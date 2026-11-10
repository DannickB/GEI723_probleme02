"""
GEI723 – Automne 2026 – Problème 2, partie II
Classification MNIST par STDP (adaptation de Diehl & Cook, 2015)

Version compatible GPU (Brian2CUDA) et C++ standalone.

En mode « standalone », Brian2 génère et compile UN programme qui simule tout
d'un coup : on ne peut plus exécuter de Python entre deux images. Donc :
  - les images sont fournies par des TimedArray (au lieu de changer
    input_group.rates dans une boucle) ;
  - la normalisation des poids se fait dans la simulation (variable
    « summed » + run_regularly) ;
  - le comptage des décharges par image se fait APRÈS la simulation, à
    partir des temps de décharge enregistrés.
Cette structure fonctionne aussi en mode 'runtime' (et y est déjà beaucoup
plus rapide qu'une boucle de net.run()).

Organisation :
    1. PARAMÈTRES             (dont DEVICE : 'runtime', 'cpp_standalone', 'cuda_standalone')
    2. charger_donnees()      MNIST → X_train, y_train, X_test, y_test
    3. creer_stimulus()       séquence d'images → TimedArray
    4. creer_neurones()       couches excitatrice et inhibitrice
    5. creer_synapses()       entrée→E (STDP éq. 3), E→I, I→E, normalisation
    6. creer_reseau()         assemble tout
    7. simuler()              entraînement, étiquetage gelé, validation, test
                              en une seule simulation
    8. compter_decharges()    décharges par (image, neurone)
    9. etiqueter_neurones()   classe de chaque neurone (réponse moyenne max)
   10. evaluer()              précision (vote : moyenne des neurones par classe)
   11. afficher_champs_recepteurs()
   12. sauvegarder_reseau() / charger_reseau()
   13. main()

Sauvegarde / rechargement : à la fin de chaque exécution, le réseau appris
(connectivité, poids, délais, seuils adaptatifs theta) et les résultats sont
écrits dans resultats/*.npz (SAUVEGARDER). Pour repartir d'un réseau appris,
mettre son chemin dans CHARGER :
  - ENTRAINER = True  : poursuit l'apprentissage (époques supplémentaires) ;
  - ENTRAINER = False : poids gelés ; les images d'entraînement servent
                        seulement à étiqueter les neurones, puis évaluation.
"""
import os
import threading
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from brian2 import *

# =============================================================================
# 1. PARAMÈTRES
# =============================================================================
# Les équations Brian2 lisent ces constantes directement (variables globales
# du module) : ne pas les renommer sans modifier les équations.

DEVICE = 'cpp_standalone'   # 'runtime' | 'cpp_standalone' | 'cuda_standalone'
# Threads OpenMP en cpp_standalone (0 = un seul thread). Par défaut : un par
# cœur physique (os.cpu_count() compte les cœurs logiques, soit 2x avec SMT).
N_THREADS = 10

SEED = 42
N_SAMPLES = 70000          # sous-ensemble de MNIST (70 000 au total)
FRAC_TEST = 0.10
FRAC_VALID = 0.10
N_EPOCHS = 2

# Sauvegarde / rechargement du réseau appris
DOSSIER_RESULTATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats')
SAUVEGARDER = True        # écrire resultats/reseau_*.npz à la fin
CHARGER = None            # chemin d'un .npz (absolu ou relatif à ce fichier) ; None = poids aléatoires
ENTRAINER = True          # False : plasticité et homéostasie coupées (évaluation d'un réseau chargé)
# Après l'entraînement, rejouer les images d'entraînement poids GELÉS pour
# étiqueter les neurones (comme Diehl & Cook) : un seul run donne alors un
# résultat final. Coûte une passe de plus sur l'entraînement (~+45 min à 7000).
# False : étiquetage avec les décharges enregistrées pendant l'apprentissage.
ETIQUETAGE_GELE = True

N_INPUT = 28 * 28         # un neurone d'entrée par pixel
N_NEURONS = 200        # neurones excitateurs (= inhibiteurs) ; Diehl & Cook : 100 à 6400

P_EI = 1.0                # connection probability E->I (1.0 = one-to-one, as in Diehl & Cook)
P_IE = 1.0                # connection probability I->E (1.0 = all E except the partner)

NORMALISER = True         # normalisation divisive des poids d'entrée
norm_target = 78.         # somme visée des poids entrant dans chaque neurone

defaultclock.dt = 0.5 * ms

time_per_sample = 0.35 * second
resting_time = 0.15 * second
T_img = time_per_sample + resting_time     # durée totale d'une image
dt_on = 50 * ms                            # résolution de l'alternance image/repos

# Neurones
v_rest_e = -65. * mV
v_rest_i = -60. * mV
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms

# Homéostasie (seuil adaptatif)
tc_theta = 1e7 * ms
theta_plus_e = 0.025 * mV     # Diehl & Cook: 0.05 mV (they also re-present images with <5 spikes)

# STDP (Diehl & Cook, éq. 3)
tc_pre_ee = 20 * ms
nu_ee_post = 0.01
wmax = 1.0
mu = 0.2
xtar = 0.3


def configurer_device():
    """Choisit le mode d'exécution. Doit être appelé AVANT de créer les objets."""
    if DEVICE == 'runtime':
        prefs.codegen.target = 'cython'   # 'numpy' si pas de compilateur C++
    elif DEVICE == 'cpp_standalone':
        set_device('cpp_standalone', build_on_run=False)
        prefs.devices.cpp_standalone.openmp_threads = N_THREADS
    elif DEVICE == 'cuda_standalone':
        import brian2cuda  # noqa: F401  (pip install brian2cuda)
        set_device('cuda_standalone', build_on_run=False)
    else:
        raise ValueError(DEVICE)


# =============================================================================
# 2. DONNÉES
# =============================================================================
def charger_donnees(n_samples=N_SAMPLES, frac_test=FRAC_TEST, frac_valid=FRAC_VALID):
    """Découpage entraînement / validation / test (stratifié par classe)."""
    X_all, y_all = datasets.fetch_openml('mnist_784', version=1,
                                         return_X_y=True, as_frame=False,
                                         parser='liac-arff')
    X, y = X_all[:n_samples].astype(float), y_all[:n_samples].astype(int)

    X_train, X_temp, y_train, y_temp = model_selection.train_test_split(
        X, y, test_size=frac_test + frac_valid, random_state=SEED, stratify=y)
    X_valid, X_test, y_valid, y_test = model_selection.train_test_split(
        X_temp, y_temp, test_size=frac_test / (frac_test + frac_valid),
        random_state=SEED, stratify=y_temp)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def afficher_echantillon(X, y, index=None):
    """Affiche une image MNIST (vecteur de 784 → matrice 28x28)."""
    if index is None:
        index = np.random.randint(0, len(X))
    plt.figure()
    plt.axis('off')
    plt.imshow(X[index].reshape(28, 28), cmap=plt.cm.gray_r)
    plt.title(f"Échantillon MNIST avec étiquette {y[index]}")
    plt.show()


# =============================================================================
# 3. STIMULUS (encodage)
# =============================================================================
def encoder_image(sample):
    """Pixels 0–255 → fréquences (Hz, sans unité ici). Max ≈ 63.75 Hz."""
    return sample / 4


def creer_stimulus(images):
    """Construit les TimedArray qui jouent la séquence d'images.

    stimulus(t, i) : fréquence du pixel i de l'image en cours (1 ligne / image)
    on(t)          : 1 pendant la présentation, 0 pendant le repos
    """
    n_images = len(images)
    taux = np.array([encoder_image(s) for s in images])          # (n_images, 784)
    stimulus = TimedArray(taux * Hz, dt=T_img)

    n_on = int(round(time_per_sample / dt_on))
    n_off = int(round(resting_time / dt_on))
    motif = np.array([1.] * n_on + [0.] * n_off)
    on = TimedArray(np.tile(motif, n_images), dt=dt_on)
    return stimulus, on


# =============================================================================
# 4. NEURONES
# =============================================================================
# Les courants synaptiques sont écrits directement dans dv/dt (les
# sous-expressions provoquaient un RecursionError dans Synapses()).
# sum_w : somme des poids entrants, calculée par les synapses (normalisation).
neuron_model = '''
    homeostasie : boolean (shared)
    dv/dt = ((v_rest - v) + ge * (0*mV - v) + gi * (E_inh - v)) / tau : volt (unless refractory)
    dge/dt = -ge / (1.0*ms)          : 1
    dgi/dt = -gi / (2.0*ms)          : 1
    dtheta/dt = -int(homeostasie) * theta / tc_theta : volt  # frozen thresholds when homeostasis is off
    sum_w  : 1
    tau    : second (constant, shared)
    v_rest : volt   (constant, shared)
    E_inh  : volt   (constant, shared)
'''


def creer_neurones(n_neurons=N_NEURONS, etat=None):
    """Crée la couche excitatrice (avec homéostasie) et la couche inhibitrice.

    etat : dictionnaire chargé par charger_reseau() (theta appris) ou None.
    """
    excitatory_group = NeuronGroup(
        n_neurons, neuron_model, refractory=refrac_e,
        threshold='v >= v_thresh_e + theta',
        reset='v = v_reset_e; theta += int(homeostasie) * theta_plus_e',
        method='euler', name='excitateurs')
    excitatory_group.homeostasie = True
    excitatory_group.tau = 100 * ms
    excitatory_group.v_rest = v_rest_e
    excitatory_group.E_inh = -100. * mV
    excitatory_group.v = v_rest_e
    if etat is not None:
        excitatory_group.theta = etat['theta'] * volt   # seuils adaptatifs appris

    inhibitory_group = NeuronGroup(
        n_neurons, neuron_model, refractory=refrac_i,
        threshold='v >= v_thresh_i',
        reset='v = v_reset_i',
        method='euler', name='inhibiteurs')
    inhibitory_group.homeostasie = False
    inhibitory_group.tau = 10 * ms
    inhibitory_group.v_rest = v_rest_i
    inhibitory_group.E_inh = -85. * mV
    inhibitory_group.v = v_rest_i

    return excitatory_group, inhibitory_group


# =============================================================================
# 5. SYNAPSES
# =============================================================================
synapse_model = "w : 1"

# STDP de Diehl & Cook, éq. 3 : Δw = η (x_pre − x_tar)(w_max − w)^μ
# Une seule trace présynaptique ; mise à jour au spike postsynaptique.
stdp_synapse_model = '''
    w : 1
    plastic : boolean (shared)    # activer/désactiver la plasticité
    dpre/dt = -pre / tc_pre_ee : 1 (event-driven)
    sum_w_post = w : 1 (summed)   # somme des poids entrants de chaque neurone E
'''

stdp_pre = '''
    ge_post += w
    pre = 1.
'''

stdp_post = '''
    w = clip(w + int(plastic) * nu_ee_post * (pre - xtar) * clip(wmax - w, 0, wmax)**mu, 0, wmax)
'''


def creer_synapses(input_group, excitatory_group, inhibitory_group, etat=None):
    """Crée les trois populations de synapses.

    etat : dictionnaire chargé par charger_reseau() ou None. S'il est fourni, la
    connectivité (i, j), les poids et les délais sont repris tels quels, ce qui
    reconstruit exactement le réseau sauvegardé.
    """
    # Entrée → excitateurs : plastique (STDP)
    input_synapse = Synapses(input_group, excitatory_group,
                             model=stdp_synapse_model,
                             on_pre=stdp_pre, on_post=stdp_post,
                             name='entree_vers_E')
    if etat is None:
        input_synapse.connect(True)           # tout-à-tout
        input_synapse.delay = 'rand()*10*ms'
        input_synapse.w = 'rand()*0.3'
    else:
        input_synapse.connect(i=etat['entree_i'], j=etat['entree_j'])
        input_synapse.delay = etat['entree_delai'] * second
        input_synapse.w = etat['entree_w']
    input_synapse.plastic = True

    # Normalisation divisive DANS la simulation : à la fin de chaque période
    # image + repos, chaque colonne est ramenée à norm_target.
    # (Pendant le test, plastic = False → les poids ne changent pas.)
    if NORMALISER:
        input_synapse.run_regularly(
            '''facteur = 1 + int(plastic) * (norm_target / (sum_w_post + 1e-12) - 1)
               w = clip(w * facteur, 0, wmax)''',
            dt=T_img, when='end', name='normalisation')

    # Excitateurs → inhibiteurs : synapse EXCITATRICE (ge)
    e_i_synapse = Synapses(excitatory_group, inhibitory_group,
                           model=synapse_model, on_pre='ge_post += w',
                           name='E_vers_I')
    if etat is None:
        e_i_synapse.connect(condition='i == j', p=P_EI)
        e_i_synapse.w = '10.4'
    else:
        e_i_synapse.connect(i=etat['ei_i'], j=etat['ei_j'])
        e_i_synapse.w = etat['ei_w']

    # Inhibiteurs → excitateurs : synapse INHIBITRICE (gi)
    i_e_synapse = Synapses(inhibitory_group, excitatory_group,
                           model=synapse_model, on_pre='gi_post += w',
                           name='I_vers_E')
    if etat is None:
        i_e_synapse.connect(condition='i != j', p=P_IE)
        i_e_synapse.w = '17.0'
    else:
        i_e_synapse.connect(i=etat['ie_i'], j=etat['ie_j'])
        i_e_synapse.w = etat['ie_w']

    return input_synapse, e_i_synapse, i_e_synapse


# =============================================================================
# 6. RÉSEAU
# =============================================================================
def creer_reseau(images, n_neurons=N_NEURONS, etat=None):
    """Construit le réseau ; `images` = toutes les images dans l'ordre joué
    (entraînement puis test). etat : réseau chargé (charger_reseau) ou None."""
    stimulus, on = creer_stimulus(images)
    input_group = PoissonGroup(N_INPUT, rates='stimulus(t, i) * on(t)',
                               namespace={'stimulus': stimulus, 'on': on},
                               name='entree')

    excitatory_group, inhibitory_group = creer_neurones(n_neurons, etat)
    input_synapse, e_i_synapse, i_e_synapse = creer_synapses(
        input_group, excitatory_group, inhibitory_group, etat)

    e_monitor = SpikeMonitor(excitatory_group)     # temps et indices des décharges

    net = Network(input_group, excitatory_group, inhibitory_group,
                  input_synapse, e_i_synapse, i_e_synapse, e_monitor)

    return dict(net=net, input_group=input_group,
                excitatory_group=excitatory_group,
                inhibitory_group=inhibitory_group,
                input_synapse=input_synapse, e_i_synapse=e_i_synapse,
                i_e_synapse=i_e_synapse, e_monitor=e_monitor)


# =============================================================================
# 7. SIMULATION
# =============================================================================
class BarreProgression:
    """Barre tqdm en nombre d'images, alimentée par le temps simulé.

    En standalone, la simulation tourne dans un programme C++ séparé : un fil
    Python lit sa sortie (lignes « PROGRESS <t simulé en s> ») pendant qu'il tourne.
    """

    def __init__(self, phases):
        """phases : liste de (nom, n_images, plastique) dans l'ordre joué."""
        self.noms = [nom for nom, _, _ in phases]
        self.fins = np.cumsum([n for _, n, _ in phases])   # image de fin de chaque phase
        self.barre = tqdm(total=int(self.fins[-1]), unit='img', dynamic_ncols=True,
                          desc='Compilation' if DEVICE != 'runtime' else 'Simulation')
        self._fil, self._arret = None, threading.Event()
        self._demarree = False

    def maj(self, t_simule):
        """Place la barre à l'image en cours pour un temps simulé (s)."""
        if not self._demarree:
            # Redémarre le chronomètre : la vitesse et le temps restant ne
            # doivent pas compter la compilation.
            self._demarree = True
            self.barre.reset(total=self.barre.total)
        n = min(int(t_simule / float(T_img / second)), self.barre.total)
        self.barre.set_description('Simulation', refresh=False)
        k = min(int(np.searchsorted(self.fins, n, side='right')), len(self.noms) - 1)
        self.barre.set_postfix_str(self.noms[k], refresh=False)
        self.barre.update(n - self.barre.n)

    def suivre_fichier(self, chemin):
        """Démarre le fil qui lit les lignes PROGRESS écrites dans `chemin`."""
        # Chemin absolu : Brian2 change de dossier courant pendant l'exécution.
        chemin = os.path.abspath(chemin)
        if os.path.exists(chemin):
            os.remove(chemin)            # sortie d'une exécution précédente
        self._fil = threading.Thread(target=self._lire, args=(chemin,), daemon=True)
        self._fil.start()

    def _lire(self, chemin):
        position, reste = 0, ''
        while True:
            fini = self._arret.is_set()   # dernière lecture complète après l'arrêt
            if os.path.exists(chemin):
                with open(chemin, errors='replace') as f:
                    f.seek(position)
                    texte = reste + f.read()
                    position = f.tell()
                *lignes, reste = texte.split('\n')   # garde une ligne incomplète
                for ligne in lignes:
                    if ligne.startswith('PROGRESS '):
                        self.maj(float(ligne.split()[1]))
                    elif ligne.strip():
                        self.barre.write(ligne)       # autres messages du programme
            if fini:
                return
            self._arret.wait(0.5)

    def arreter(self):
        if self._fil is not None:
            self._arret.set()
            self._fil.join()

    def fermer(self):
        self.barre.close()


def simuler(reseau, phases):
    """Joue les phases dans l'ordre, en une seule simulation.

    phases : liste de (nom, n_images, plastique). plastique=True active la STDP
    et l'homéostasie ; False gèle les poids et les seuils (étiquetage, test).
    """
    net = reseau['net']
    barre = BarreProgression(phases)

    # Rapport de progression : appel Python en runtime ; en standalone, le
    # programme C++ écrit « PROGRESS <t simulé> » que BarreProgression lit.
    if DEVICE == 'runtime':
        rapport = lambda elapsed, completed, start, duration: barre.maj(
            float(start) + completed * float(duration))
    else:
        rapport = ('std::cout << "PROGRESS " << start + completed*duration'
                   ' << std::endl << std::flush;')

    for nom, n_images, plastique in phases:
        if n_images == 0:
            continue
        print(f"{nom.capitalize()} : {n_images} images ({n_images * T_img}), "
              + ("plastique" if plastique else "poids gelés"))
        reseau['input_synapse'].plastic = plastique
        reseau['excitatory_group'].homeostasie = plastique
        net.run(n_images * T_img, report=rapport, report_period=1 * second)

    if DEVICE != 'runtime':
        # Génère, compile et exécute le programme complet (CPU C++ ou GPU).
        # with_output=False : la sortie du programme va dans results/stdout.txt,
        # que BarreProgression suit pendant l'exécution.
        dossier = f'build_{DEVICE}'
        barre.suivre_fichier(os.path.join(dossier, 'results', 'stdout.txt'))
        try:
            device.build(directory=dossier, compile=True, run=True, with_output=False)
        finally:
            barre.arreter()
    barre.fermer()

    print("Nombre de synapses :",
          len(reseau['input_synapse']) + len(reseau['e_i_synapse'])
          + len(reseau['i_e_synapse']))


# =============================================================================
# 8. COMPTAGE DES DÉCHARGES
# =============================================================================
def compter_decharges(e_monitor, n_images, n_neurons=N_NEURONS):
    """counts[k, n] = décharges du neurone n pendant la présentation de l'image k
    (les décharges pendant le repos sont ignorées)."""
    t = np.asarray(e_monitor.t / second)
    idx = np.asarray(e_monitor.i)
    T = float(T_img / second)

    image = (t // T).astype(int)
    pendant_image = (t - image * T) < float(time_per_sample / second)
    garder = pendant_image & (image < n_images)

    counts = np.zeros((n_images, n_neurons))
    np.add.at(counts, (image[garder], idx[garder]), 1)
    return counts


# =============================================================================
# 9. ÉTIQUETAGE
# =============================================================================
def etiqueter_neurones(counts_train, y_train):
    """Diehl & Cook: each neuron gets the class with its highest mean response."""
    reponse_moyenne = np.zeros((10, counts_train.shape[1]))
    for c in range(10):
        if np.any(y_train == c):     # une classe absente donnerait NaN, qui gagnerait l'argmax
            reponse_moyenne[c] = counts_train[y_train == c].mean(axis=0)
    assignations = np.argmax(reponse_moyenne, axis=0)        # (n_neurons,)
    assignations[reponse_moyenne.max(axis=0) == 0] = -1      # never fired: no class
    print("Neurones par classe :", np.bincount(assignations[assignations >= 0], minlength=10))
    return assignations, reponse_moyenne


# =============================================================================
# 10. ÉVALUATION
# =============================================================================
def evaluer(counts, y, assignations):
    """Prediction = class whose assigned neurons have the highest mean spike count."""
    scores = np.zeros((len(counts), 10))
    for c in range(10):
        neurones = assignations == c
        if neurones.any():
            scores[:, c] = counts[:, neurones].mean(axis=1)
    predictions = np.argmax(scores, axis=1)
    accuracy = np.mean(predictions == y)
    print(f"Précision du modèle : {accuracy:.3f}")
    print(f"Images sans aucune décharge : {np.mean(counts.sum(axis=1) == 0):.1%}")
    return accuracy, predictions


# =============================================================================
# 11. VISUALISATION
# =============================================================================
def afficher_champs_recepteurs(input_synapse, n_neurons=N_NEURONS):
    """Poids appris de chaque neurone E, sous forme d'images 28x28."""
    W = np.zeros((N_INPUT, n_neurons))
    W[np.asarray(input_synapse.i), np.asarray(input_synapse.j)] = np.asarray(input_synapse.w)
    cote = int(np.ceil(np.sqrt(n_neurons)))
    grille = np.zeros((cote * 28, cote * 28))
    for n in range(n_neurons):
        r, c = divmod(n, cote)
        grille[r*28:(r+1)*28, c*28:(c+1)*28] = W[:, n].reshape(28, 28)
    plt.figure(figsize=(8, 8))
    plt.imshow(grille, cmap='hot_r')
    plt.axis('off')
    plt.title('Champs récepteurs (poids entrée → E)')
    plt.colorbar()
    plt.show()


# =============================================================================
# 12. SAUVEGARDE / RECHARGEMENT
# =============================================================================
def sauvegarder_reseau(reseau, resultats, epoques_totales):
    """Écrit le réseau appris et les résultats dans resultats/reseau_*.npz.

    Contenu : connectivité, poids et délais des trois synapses, theta des
    neurones E (de quoi reconstruire exactement le réseau), plus les décomptes
    de décharges, étiquettes et précisions (pour les analyses des parties III/IV).
    """
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    horodatage = time.strftime('%Y%m%d_%H%M%S')
    chemin = os.path.join(
        DOSSIER_RESULTATS,
        f'reseau_{N_NEURONS}n_{epoques_totales}ep_{N_SAMPLES}img_{horodatage}.npz')

    s_in, s_ei, s_ie = (reseau['input_synapse'], reseau['e_i_synapse'],
                        reseau['i_e_synapse'])
    np.savez_compressed(
        chemin,
        # Réseau (rechargé par charger_reseau)
        entree_i=np.asarray(s_in.i), entree_j=np.asarray(s_in.j),
        entree_w=np.asarray(s_in.w), entree_delai=np.asarray(s_in.delay / second),
        ei_i=np.asarray(s_ei.i), ei_j=np.asarray(s_ei.j), ei_w=np.asarray(s_ei.w),
        ie_i=np.asarray(s_ie.i), ie_j=np.asarray(s_ie.j), ie_w=np.asarray(s_ie.w),
        theta=np.asarray(reseau['excitatory_group'].theta / volt),
        # Configuration (vérifiée au rechargement)
        n_input=N_INPUT, n_neurons=N_NEURONS, seed=SEED, n_samples=N_SAMPLES,
        frac_test=FRAC_TEST, frac_valid=FRAC_VALID, epoques=epoques_totales,
        # Résultats
        **resultats)
    print(f"Réseau sauvegardé : {chemin}")
    return chemin


def charger_reseau(chemin):
    """Lit un .npz écrit par sauvegarder_reseau() et vérifie sa compatibilité."""
    if not os.path.isabs(chemin):
        chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), chemin)
    with np.load(chemin) as f:
        etat = {k: f[k] for k in f.files}

    if int(etat['n_neurons']) != N_NEURONS or int(etat['n_input']) != N_INPUT:
        raise ValueError(
            f"Réseau sauvegardé avec {int(etat['n_neurons'])} neurones et "
            f"{int(etat['n_input'])} entrées ; N_NEURONS={N_NEURONS}, N_INPUT={N_INPUT}.")
    # Un découpage différent mettrait des images d'entraînement dans le test.
    if (int(etat['seed']) != SEED or int(etat['n_samples']) != N_SAMPLES
            or float(etat['frac_test']) != FRAC_TEST
            or float(etat['frac_valid']) != FRAC_VALID):
        print("ATTENTION : SEED / N_SAMPLES / FRAC_* diffèrent de ceux de "
              "l'entraînement sauvegardé ; les ensembles de validation et de test "
              "peuvent contenir des images déjà apprises.")
    print(f"Réseau chargé : {chemin} ({int(etat['epoques'])} époque(s) apprise(s))")
    return etat


# =============================================================================
# 13. PROGRAMME PRINCIPAL
# =============================================================================
def main():
    configurer_device()
    np.random.seed(SEED)
    seed(SEED)

    X_train, y_train, X_valid, y_valid, X_test, y_test = charger_donnees()

    etat = charger_reseau(CHARGER) if CHARGER else None
    epoques_precedentes = int(etat['epoques']) if etat is not None else 0
    n_epochs = N_EPOCHS if ENTRAINER else 0
    epoques_totales = epoques_precedentes + n_epochs
    # Sans entraînement, la passe gelée sur l'entraînement est toujours nécessaire.
    etiquetage_gele = ETIQUETAGE_GELE or not ENTRAINER

    # Séquence jouée, en une seule simulation :
    #   1. entraînement (plastique), N_EPOCHS fois
    #   2. étiquetage : images d'entraînement, poids gelés
    #   3. validation, puis test (poids gelés)
    n_tr = len(X_train)
    phases = [('entraînement', n_epochs * n_tr, True),
              ('étiquetage', n_tr if etiquetage_gele else 0, False),
              ('validation', len(X_valid), False),
              ('test', len(X_test), False)]
    X_seq = np.concatenate([np.tile(X_train, (n_epochs, 1)),
                            X_train[:phases[1][1]], X_valid, X_test])

    # N_NEURONS passed explicitly: default arguments are frozen at import time,
    # so a value overridden later (e.g. by serie_entrainements.py) would be ignored.
    reseau = creer_reseau(X_seq, N_NEURONS, etat=etat)
    simuler(reseau, phases)

    # Découpe des décharges par phase
    counts = compter_decharges(reseau['e_monitor'], len(X_seq), N_NEURONS)
    debuts = np.concatenate([[0], np.cumsum([n for _, n, _ in phases])])
    par_phase = {nom: counts[debuts[k]:debuts[k + 1]]
                 for k, (nom, _, _) in enumerate(phases)}
    counts_valid, counts_test = par_phase['validation'], par_phase['test']
    if etiquetage_gele:
        counts_etiq = par_phase['étiquetage']
        print("Étiquetage : décharges de la passe poids gelés")
    else:
        counts_etiq = par_phase['entraînement'][-n_tr:]   # dernière époque
        print("Étiquetage : décharges enregistrées pendant l'apprentissage")

    assignations, _ = etiqueter_neurones(counts_etiq, y_train)
    # Précision sur l'entraînement (optimiste : mêmes images que l'étiquetage)
    print("Entraînement :");
    acc_train, _ = evaluer(counts_etiq, y_train, assignations)
    print("Validation :");
    acc_valid, _ = evaluer(counts_valid, y_valid, assignations)
    print("Test :");
    acc_test, _ = evaluer(counts_test, y_test, assignations)

    if SAUVEGARDER:
        sauvegarder_reseau(reseau, dict(
            assignations=assignations, etiquetage_gele=etiquetage_gele,
            acc_train=acc_train, acc_valid=acc_valid, acc_test=acc_test,
            counts_apprentissage=par_phase['entraînement'][-n_tr:].astype(np.uint16),
            counts_etiquetage=counts_etiq.astype(np.uint16), y_train=y_train,
            counts_valid=counts_valid.astype(np.uint16), y_valid=y_valid,
            counts_test=counts_test.astype(np.uint16), y_test=y_test,
        ), epoques_totales)

    afficher_champs_recepteurs(reseau['input_synapse'], N_NEURONS)


if __name__ == '__main__':
    main()
