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
    7. simuler()              entraînement puis test en une seule simulation
    8. compter_decharges()    décharges par (image, neurone)
    9. etiqueter_neurones()   meilleur neurone par classe
   10. evaluer()              précision
   11. afficher_champs_recepteurs()
   12. main()
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from brian2 import *

# =============================================================================
# 1. PARAMÈTRES
# =============================================================================
# Les équations Brian2 lisent ces constantes directement (variables globales
# du module) : ne pas les renommer sans modifier les équations.

DEVICE = 'cpp_standalone'   # 'runtime' | 'cpp_standalone' | 'cuda_standalone'

SEED = 0
N_SAMPLES = 7000          # sous-ensemble de MNIST (70 000 au total)
FRAC_TEST = 0.10
FRAC_VALID = 0.10
N_EPOCHS = 1

N_INPUT = 28 * 28         # un neurone d'entrée par pixel
N_NEURONS = 28*28           # neurones excitateurs (= inhibiteurs) ; Diehl & Cook : 100 à 6400

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
theta_plus_e = 0.05 * mV

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
    dtheta/dt = -theta / tc_theta    : volt
    sum_w  : 1
    tau    : second (constant, shared)
    v_rest : volt   (constant, shared)
    E_inh  : volt   (constant, shared)
'''


def creer_neurones(n_neurons=N_NEURONS):
    """Crée la couche excitatrice (avec homéostasie) et la couche inhibitrice."""
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


def creer_synapses(input_group, excitatory_group, inhibitory_group):
    """Crée les trois populations de synapses."""
    # Entrée → excitateurs : plastique (STDP)
    input_synapse = Synapses(input_group, excitatory_group,
                             model=stdp_synapse_model,
                             on_pre=stdp_pre, on_post=stdp_post,
                             name='entree_vers_E')
    input_synapse.connect(True)               # tout-à-tout
    input_synapse.delay = 'rand()*10*ms'
    input_synapse.plastic = True
    input_synapse.w = 'rand()*0.3'

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
    e_i_synapse.connect(True, p=0.0025)
    e_i_synapse.w = 'rand()*10.4'

    # Inhibiteurs → excitateurs : synapse INHIBITRICE (gi)
    i_e_synapse = Synapses(inhibitory_group, excitatory_group,
                           model=synapse_model, on_pre='gi_post += w',
                           name='I_vers_E')
    i_e_synapse.connect(True, p=0.9)
    i_e_synapse.w = 'rand()*17.0'

    return input_synapse, e_i_synapse, i_e_synapse


# =============================================================================
# 6. RÉSEAU
# =============================================================================
def creer_reseau(images, n_neurons=N_NEURONS):
    """Construit le réseau ; `images` = toutes les images dans l'ordre joué
    (entraînement puis test)."""
    stimulus, on = creer_stimulus(images)
    input_group = PoissonGroup(N_INPUT, rates='stimulus(t, i) * on(t)',
                               namespace={'stimulus': stimulus, 'on': on},
                               name='entree')

    excitatory_group, inhibitory_group = creer_neurones(n_neurons)
    input_synapse, e_i_synapse, i_e_synapse = creer_synapses(
        input_group, excitatory_group, inhibitory_group)

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
def simuler(reseau, n_train, n_test):
    """Entraînement (plasticité + homéostasie) puis test (tout gelé)."""
    net = reseau['net']

    print(f"Entraînement : {n_train} images ({n_train * T_img})")
    reseau['input_synapse'].plastic = True
    reseau['excitatory_group'].homeostasie = True
    net.run(n_train * T_img, report='text')

    print(f"Test : {n_test} images ({n_test * T_img})")
    reseau['input_synapse'].plastic = False
    reseau['excitatory_group'].homeostasie = False
    net.run(n_test * T_img, report='text')

    if DEVICE != 'runtime':
        # Génère, compile et exécute le programme complet (CPU C++ ou GPU)
        device.build(directory=f'build_{DEVICE}', compile=True, run=True)

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
    """Pour chaque classe, le neurone qui a le plus déchargé."""
    spikes = np.zeros((10, counts_train.shape[1]))
    for c in range(10):
        spikes[c] = counts_train[y_train == c].sum(axis=0)
    labeled_neurons = np.argmax(spikes, axis=1)
    print("Neurones étiquetés (classe → neurone) :", labeled_neurons)
    return labeled_neurons, spikes


# =============================================================================
# 10. ÉVALUATION
# =============================================================================
def evaluer(counts_test, y_test, labeled_neurons):
    """Prédiction = classe dont le neurone étiqueté a le plus déchargé."""
    predictions = np.argmax(counts_test[:, labeled_neurons], axis=1)
    accuracy = np.mean(predictions == y_test)
    silencieux = np.mean(counts_test.sum(axis=1) == 0)
    print(f"Précision du modèle : {accuracy:.3f}")
    print(f"Images sans aucune décharge : {silencieux:.1%}")
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
# 12. PROGRAMME PRINCIPAL
# =============================================================================
def main():
    configurer_device()
    np.random.seed(SEED)
    seed(SEED)

    X_train, y_train, X_valid, y_valid, X_test, y_test = charger_donnees()

    # Séquence jouée : entraînement (plastique), puis validation et test (gelés)
    X_seq = np.concatenate([np.tile(X_train, (N_EPOCHS, 1)), X_valid, X_test])
    n_train = N_EPOCHS * len(X_train)
    n_valid, n_test = len(X_valid), len(X_test)

    reseau = creer_reseau(X_seq)
    simuler(reseau, n_train, n_valid + n_test)

    counts = compter_decharges(reseau['e_monitor'], n_train + n_valid + n_test)
    counts_train = counts[:n_train][-len(X_train):]  # dernière époque
    counts_valid = counts[n_train:n_train + n_valid]
    counts_test = counts[n_train + n_valid:]

    labeled_neurons, _ = etiqueter_neurones(counts_train, y_train)
    print("Validation :");
    evaluer(counts_valid, y_valid, labeled_neurons)
    print("Test :");
    evaluer(counts_test, y_test, labeled_neurons)
    afficher_champs_recepteurs(reseau['input_synapse'])


if __name__ == '__main__':
    main()
