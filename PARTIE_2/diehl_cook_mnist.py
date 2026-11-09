"""
GEI723 – Automne 2026 – Problème 2, partie II
Classification MNIST par STDP (adaptation de Diehl & Cook, 2015)

Version compatible GPU (Brian2CUDA) et C++ standalone.

En mode « standalone », Brian2 génère et compile UN programme qui simule tout
d'un coup : on ne peut plus exécuter de Python entre deux images. Donc :
  - les images sont fournies par des TimedArray ;
  - la normalisation des poids se fait dans la simulation (variable
    « summed » + run_regularly) ;
  - le comptage des décharges par image se fait APRÈS la simulation.

Figures et résultats :
  Figure/<configuration>/   une figure .png par analyse
  Figure/resultats.csv      une ligne par exécution (pour le tableau du rapport)

Organisation :
    1. PARAMÈTRES             (DEVICE, taille du réseau, données, sauvegarde…)
    2. charger_donnees()      MNIST → entraînement / validation / test
    3. creer_stimulus()       séquence d'images → TimedArray
    4. creer_neurones()       couches excitatrice et inhibitrice
    5. creer_synapses()       entrée→E (STDP éq. 3), E→I, I→E, normalisation
    6. creer_reseau()         assemble tout
    7. simuler()              entraînement puis évaluation en une simulation
    8. compter_decharges()    décharges par (image, neurone)
    9. etiqueter_neurones()   classe assignée à chaque neurone
   10. evaluer()              précision, images silencieuses
   11. FIGURES                sauvegarde dans Figure/
   12. RÉSULTATS              ajout d'une ligne dans Figure/resultats.csv
   13. main()
"""
from brian2 import *
# Imports de la bibliothèque standard APRÈS « from brian2 import * » :
# Brian2 exporte aussi des noms comme « datetime » qui écraseraient les nôtres.
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

# =============================================================================
# 1. PARAMÈTRES
# =============================================================================
# Les équations Brian2 lisent ces constantes directement (variables globales
# du module) : ne pas les renommer sans modifier les équations.

DEVICE = 'cpp_standalone'   # 'runtime' | 'cpp_standalone' | 'cuda_standalone'

SEED = 0
N_SAMPLES = 60000         # sous-ensemble de MNIST (70 000 au total)
FRAC_TEST = 0.10
FRAC_VALID = 0.10
N_EPOCHS = 1

N_INPUT = 28 * 28         # un neurone d'entrée par pixel
N_NEURONS = 200           # neurones excitateurs (= inhibiteurs) ; Diehl & Cook : 100 à 6400

HOMEOSTASIE = True        # seuil adaptatif pendant l'entraînement
NORMALISER = True         # normalisation divisive des poids d'entrée
norm_target = 78.         # somme visée des poids entrant dans chaque neurone

# Sauvegarde
DOSSIER_FIGURES = Path(__file__).resolve().parent / 'Figure'
AFFICHER_FIGURES = False  # True : afficher aussi à l'écran (bloque jusqu'à fermeture)

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


def nom_configuration():
    """Nom court de la configuration, utilisé pour le sous-dossier des figures."""
    return (f"N{N_NEURONS}_ep{N_EPOCHS}_s{N_SAMPLES}"
            f"_homeo{int(HOMEOSTASIE)}_mu{mu}_xtar{xtar}")


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


# =============================================================================
# 3. STIMULUS (encodage)
# =============================================================================
def encoder_image(sample):
    """Pixels 0–255 → fréquences (Hz, sans unité ici). Max ≈ 63.75 Hz."""
    return sample / 4


def creer_stimulus(images):
    """stimulus(t, i) : fréquence du pixel i de l'image en cours (1 ligne / image)
    on(t)          : 1 pendant la présentation, 0 pendant le repos"""
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
# theta ne décroît que si homeostasie = True : pendant l'évaluation, le seuil
# reste exactement figé (sinon la validation, jouée en premier, est
# désavantagée par rapport au test).
neuron_model = '''
    homeostasie : boolean (shared)
    dv/dt = ((v_rest - v) + ge * (0*mV - v) + gi * (E_inh - v)) / tau : volt (unless refractory)
    dge/dt = -ge / (1.0*ms)          : 1
    dgi/dt = -gi / (2.0*ms)          : 1
    dtheta/dt = -int(homeostasie) * theta / tc_theta : volt
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
    excitatory_group.homeostasie = HOMEOSTASIE
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
    # Entrée → excitateurs : plastique (STDP), tout-à-tout
    input_synapse = Synapses(input_group, excitatory_group,
                             model=stdp_synapse_model,
                             on_pre=stdp_pre, on_post=stdp_post,
                             name='entree_vers_E')
    input_synapse.connect(condition=True, p=1)
    input_synapse.delay = 'rand()*10*ms'
    input_synapse.plastic = True
    input_synapse.w = 'rand()*0.3'

    # Normalisation divisive dans la simulation (facteur = 1 quand plastic = False)
    if NORMALISER:
        input_synapse.run_regularly(
            '''facteur = 1 + int(plastic) * (norm_target / (sum_w_post + 1e-12) - 1)
               w = clip(w * facteur, 0, wmax)''',
            dt=T_img, when='end', name='normalisation')

    # E → I : un-à-un (chaque neurone E a son neurone I)
    e_i_synapse = Synapses(excitatory_group, inhibitory_group,
                           model=synapse_model, on_pre='ge_post += w',
                           name='E_vers_I')
    e_i_synapse.connect(condition="i==j", p=1)
    e_i_synapse.w = 'rand()*10.4'

    # I → E : chaque I inhibe tous les E sauf le sien (winner-take-all)
    i_e_synapse = Synapses(inhibitory_group, excitatory_group,
                           model=synapse_model, on_pre='gi_post += w',
                           name='I_vers_E')
    i_e_synapse.connect(condition="i!=j", p=1)
    i_e_synapse.w = 'rand()*17.0'

    return input_synapse, e_i_synapse, i_e_synapse


# =============================================================================
# 6. RÉSEAU
# =============================================================================
def creer_reseau(images, n_neurons=N_NEURONS):
    """`images` = toutes les images dans l'ordre joué (entraînement, validation, test)."""
    stimulus, on = creer_stimulus(images)
    input_group = PoissonGroup(N_INPUT, rates='stimulus(t, i) * on(t)',
                               namespace={'stimulus': stimulus, 'on': on},
                               name='entree')

    excitatory_group, inhibitory_group = creer_neurones(n_neurons)
    input_synapse, e_i_synapse, i_e_synapse = creer_synapses(
        input_group, excitatory_group, inhibitory_group)

    e_monitor = SpikeMonitor(excitatory_group)

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
def simuler(reseau, n_train, n_eval):
    """Entraînement (plasticité + homéostasie selon HOMEOSTASIE),
    puis évaluation (validation + test) avec tout gelé."""
    net = reseau['net']

    print(f"Entraînement : {n_train} images ({n_train * T_img})")
    reseau['input_synapse'].plastic = True
    reseau['excitatory_group'].homeostasie = HOMEOSTASIE
    net.run(n_train * T_img, report='text')

    print(f"Évaluation : {n_eval} images ({n_eval * T_img})")
    reseau['input_synapse'].plastic = False
    reseau['excitatory_group'].homeostasie = False
    net.run(n_eval * T_img, report='text')

    if DEVICE != 'runtime':
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
    """Chaque neurone est assigné à la classe pour laquelle il décharge le plus (en moyenne)."""
    moyennes = np.zeros((10, counts_train.shape[1]))
    for c in range(10):
        moyennes[c] = counts_train[y_train == c].mean(axis=0)
    assignations = np.argmax(moyennes, axis=0)
    print("Neurones par classe :", np.bincount(assignations, minlength=10))
    return assignations, moyennes


# =============================================================================
# 10. ÉVALUATION
# =============================================================================
def evaluer(counts, y, assignations, nom=''):
    """Vote : classe dont les neurones assignés déchargent le plus en moyenne."""
    votes = np.zeros((len(counts), 10))
    for c in range(10):
        membres = assignations == c
        if membres.any():
            votes[:, c] = counts[:, membres].mean(axis=1)
    predictions = np.argmax(votes, axis=1)

    actif = counts.sum(axis=1) > 0
    res = dict(
        precision=np.mean(predictions == y),
        silencieux=1 - np.mean(actif),
        precision_actives=np.mean(predictions[actif] == y[actif]) if actif.any() else np.nan,
    )
    print(f"{nom:>10} : précision {res['precision']:.3f}   "
          f"actives {res['precision_actives']:.3f}   "
          f"sans décharge {res['silencieux']:.1%}")
    return res, predictions


# =============================================================================
# 11. FIGURES
# =============================================================================
def sauvegarder(fig, dossier, nom):
    """Enregistre la figure en PNG dans `dossier`, l'affiche si demandé, puis la ferme."""
    chemin = dossier / f"{nom}.png"
    fig.savefig(chemin, dpi=150, bbox_inches='tight')
    print(f"  figure : {chemin}")
    if AFFICHER_FIGURES:
        plt.show()
    plt.close(fig)


def matrice_poids(input_synapse, n_neurons=N_NEURONS):
    """Poids entrée → E sous forme de matrice (784, n_neurons)."""
    W = np.zeros((N_INPUT, n_neurons))
    W[np.asarray(input_synapse.i), np.asarray(input_synapse.j)] = np.asarray(input_synapse.w)
    return W


def fig_champs_recepteurs(W, assignations, dossier):
    """Poids appris de chaque neurone E (28x28), avec la classe assignée."""
    n = W.shape[1]
    cote = int(np.ceil(np.sqrt(n)))
    rangees = int(np.ceil(n / cote))
    fig, axes = plt.subplots(rangees, cote, figsize=(cote * 0.9, rangees * 0.95 + 0.4),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis('off')
        if k < n:
            ax.imshow(W[:, k].reshape(28, 28), cmap='hot_r', vmin=0, vmax=W.max())
            ax.set_title(str(assignations[k]), fontsize=6, pad=1)
    fig.suptitle('Champs récepteurs (classe assignée au-dessus)')
    fig.tight_layout()
    sauvegarder(fig, dossier, 'champs_recepteurs')


def en_grille(valeurs, n_neurons=N_NEURONS):
    """Range un vecteur (un nombre par neurone) dans la même grille que les
    champs récepteurs : le neurone k est au même endroit dans les deux figures."""
    cote = int(np.ceil(np.sqrt(n_neurons)))
    rangees = int(np.ceil(n_neurons / cote))
    grille = np.full(cote * rangees, np.nan)
    grille[:n_neurons] = valeurs
    return grille.reshape(rangees, cote)


def fig_courbes_accord(moyennes, assignations, dossier):
    """Courbes d'accord : réponse moyenne de chaque neurone à chaque classe.

    Gauche : tous les neurones (normalisés par leur maximum), triés par classe
             assignée puis par sélectivité.
    Droite : le neurone le plus sélectif de chaque classe.
    """
    n = moyennes.shape[1]
    norm = moyennes / np.maximum(moyennes.max(axis=0, keepdims=True), 1e-12)
    # Sélectivité : réponse à la classe préférée vs moyenne des autres classes
    pref = moyennes.max(axis=0)
    autres = (moyennes.sum(axis=0) - pref) / 9
    selectivite = (pref - autres) / np.maximum(pref + autres, 1e-12)
    ordre = np.lexsort((-selectivite, assignations))

    fig = plt.figure(figsize=(14, 5))
    ax0 = fig.add_subplot(1, 2, 1)
    im = ax0.imshow(norm[:, ordre], aspect='auto', cmap='viridis', vmin=0, vmax=1,
                    interpolation='nearest')
    ax0.set_yticks(range(10))
    ax0.set_ylabel('Classe présentée')
    ax0.set_xlabel('Neurones (triés par classe assignée)')
    frontieres = np.flatnonzero(np.diff(assignations[ordre])) + 0.5
    for f in frontieres:
        ax0.axvline(f, color='white', lw=0.6)
    ax0.set_title("Courbes d'accord (réponse / réponse max du neurone)")
    fig.colorbar(im, ax=ax0, fraction=0.04)

    ax1 = fig.add_subplot(1, 2, 2)
    for c in range(10):
        membres = np.flatnonzero(assignations == c)
        if len(membres) == 0:
            continue
        k = membres[np.argmax(selectivite[membres])]
        ax1.plot(range(10), moyennes[:, k], marker='o', ms=3,
                 label=f'classe {c} (n°{k}, S={selectivite[k]:.2f})')
    ax1.set_xticks(range(10))
    ax1.set_xlabel('Classe présentée')
    ax1.set_ylabel('Décharges moyennes par image')
    ax1.set_title('Neurone le plus sélectif de chaque classe')
    ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    sauvegarder(fig, dossier, 'courbes_accord')

    # Distribution de la sélectivité (utile pour comparer des configurations)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(selectivite[np.isfinite(selectivite)], bins=30, color='tab:green')
    ax.set_xlabel('Indice de sélectivité S (0 = aucune, 1 = une seule classe)')
    ax.set_ylabel('Nombre de neurones')
    ax.set_title(f'Sélectivité des neurones (médiane {np.nanmedian(selectivite):.2f})')
    sauvegarder(fig, dossier, 'selectivite')


def cmap_activite():
    """Carte de couleurs 'hot' où les cases sans neurone (NaN) sont grises."""
    cmap = plt.get_cmap('hot').copy()
    cmap.set_bad('#bbbbbb')
    return cmap


def fig_cartes_activation_classes(counts, y, dossier, nom):
    """Activité moyenne de la couche excitatrice pour chaque classe,
    disposée comme la grille des champs récepteurs."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    cartes = [en_grille(counts[y == c].mean(axis=0)) for c in range(10)]
    vmax = np.nanmax(cartes)
    for c, ax in enumerate(axes.flat):
        im = ax.imshow(cartes[c], cmap=cmap_activite(), vmin=0, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(f'Classe {c}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, label='Décharges moyennes par image')
    fig.suptitle(f"Cartes d'activation moyennes par classe ({nom})")
    sauvegarder(fig, dossier, f'cartes_activation_{nom}')


def fig_exemples_activation(X, counts, y, predictions, dossier, nom, n_exemples=8):
    """Quelques images et l'activité de la couche qu'elles provoquent
    (titre vert = bonne prédiction, rouge = erreur)."""
    rng = np.random.default_rng(SEED)
    choix = rng.choice(len(X), size=min(n_exemples, len(X)), replace=False)
    fig, axes = plt.subplots(2, len(choix), figsize=(1.8 * len(choix), 4))
    for col, k in enumerate(choix):
        axes[0, col].imshow(X[k].reshape(28, 28), cmap='gray_r')
        bon = predictions[k] == y[k]
        axes[0, col].set_title(f'vrai {y[k]} / prédit {predictions[k]}', fontsize=8,
                               color='green' if bon else 'red')
        axes[1, col].imshow(en_grille(counts[k]), cmap=cmap_activite(),
                            interpolation='nearest')
        axes[1, col].set_xlabel(f'{int(counts[k].sum())} décharges', fontsize=8)
        for a in axes[:, col]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_ylabel('Entrée')
    axes[1, 0].set_ylabel('Couche E')
    fig.suptitle(f"Exemples d'activation ({nom})")
    fig.tight_layout()
    sauvegarder(fig, dossier, f'exemples_activation_{nom}')


def fig_histogramme_poids(W, dossier):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(W.ravel(), bins=50, color='tab:blue')
    ax.set_yscale('log')
    ax.set_xlabel('w')
    ax.set_ylabel("Nombre de synapses (log)")
    ax.set_title('Distribution des poids entrée → E après apprentissage')
    sauvegarder(fig, dossier, 'histogramme_poids')


def fig_neurones_par_classe(assignations, dossier):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(10), np.bincount(assignations, minlength=10), color='tab:blue')
    ax.set_xticks(range(10))
    ax.set_xlabel('Classe')
    ax.set_ylabel('Nombre de neurones assignés')
    ax.set_title('Répartition des neurones par classe')
    sauvegarder(fig, dossier, 'neurones_par_classe')


def fig_matrice_confusion(y, predictions, dossier, nom):
    M = np.zeros((10, 10))
    np.add.at(M, (y, predictions), 1)
    M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)     # normalisée par ligne
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=6,
                    color='white' if M[i, j] > 0.5 else 'black')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Vraie classe')
    ax.set_title(f'Matrice de confusion ({nom})')
    fig.colorbar(im, ax=ax)
    sauvegarder(fig, dossier, f'confusion_{nom}')


def fig_activite(counts, n_train, n_valid, dossier, fenetre=200):
    """Décharges par image et proportion d'images silencieuses (moyenne glissante)."""
    total = counts.sum(axis=1)
    fenetre = max(1, min(fenetre, len(total)))
    noyau = np.ones(fenetre) / fenetre
    lisse = np.convolve(total, noyau, mode='valid')          # pas d'effet de bord
    silence = np.convolve((total == 0).astype(float), noyau, mode='valid')
    x = np.arange(len(lisse)) + fenetre // 2                 # centre de chaque fenêtre

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(x, lisse, color='tab:blue')
    ax[0].set_ylabel('Décharges / image')
    ax[1].plot(x, silence * 100, color='tab:red')
    ax[1].set_ylabel('Images silencieuses (%)')
    ax[1].set_xlabel('Index de l\'image présentée')
    for a in ax:
        a.axvline(n_train, color='k', ls='--', lw=1)
        a.axvline(n_train + n_valid, color='k', ls=':', lw=1)
    ax[0].set_title(f'Activité du réseau (moyenne glissante sur {fenetre} images ; '
                    f'-- fin entraînement, ·· début test)')
    sauvegarder(fig, dossier, 'activite')


def fig_theta(excitatory_group, dossier):
    theta = np.asarray(excitatory_group.theta / mV)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(theta, bins=30, color='tab:orange')
    ax.set_xlabel('θ (mV)')
    ax.set_ylabel('Nombre de neurones')
    ax.set_title('Seuil adaptatif θ en fin d\'entraînement')
    sauvegarder(fig, dossier, 'theta')


# =============================================================================
# 12. RÉSULTATS (CSV)
# =============================================================================
def enregistrer_resultats(res_valid, res_test, duree_s, n_train):
    """Ajoute une ligne à Figure/resultats.csv (crée l'en-tête au besoin)."""
    fichier = DOSSIER_FIGURES / 'resultats.csv'
    ligne = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'configuration': nom_configuration(),
        'epoques': N_EPOCHS,
        'neurones': N_NEURONS,
        'homeostasie': HOMEOSTASIE,
        'normalisation': NORMALISER,
        'mu': mu,
        'xtar': xtar,
        'n_train': n_train,
        'seed': SEED,
        'device': DEVICE,
        'temps_min': round(duree_s / 60, 1),
        'valid': round(res_valid['precision'], 3),
        'valid_actives': round(res_valid['precision_actives'], 3),
        'valid_silence': round(res_valid['silencieux'], 3),
        'test': round(res_test['precision'], 3),
        'test_actives': round(res_test['precision_actives'], 3),
        'test_silence': round(res_test['silencieux'], 3),
    }
    nouveau = not fichier.exists()
    with open(fichier, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(ligne), delimiter=';')
        if nouveau:
            writer.writeheader()
        writer.writerow(ligne)
    print(f"  résultats ajoutés à : {fichier}")


# =============================================================================
# 13. PROGRAMME PRINCIPAL
# =============================================================================
def main():
    debut = time.perf_counter()
    configurer_device()
    np.random.seed(SEED)
    seed(SEED)

    dossier = DOSSIER_FIGURES / nom_configuration()
    dossier.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_valid, y_valid, X_test, y_test = charger_donnees()

    # Séquence jouée : entraînement (plastique), puis validation et test (gelés)
    X_seq = np.concatenate([np.tile(X_train, (N_EPOCHS, 1)), X_valid, X_test])
    n_train = N_EPOCHS * len(X_train)
    n_valid, n_test = len(X_valid), len(X_test)

    reseau = creer_reseau(X_seq)
    simuler(reseau, n_train, n_valid + n_test)

    counts = compter_decharges(reseau['e_monitor'], n_train + n_valid + n_test)
    counts_train = counts[:n_train][-len(X_train):]   # dernière époque
    counts_valid = counts[n_train:n_train + n_valid]
    counts_test = counts[n_train + n_valid:]

    assignations, moyennes = etiqueter_neurones(counts_train, y_train)
    res_valid, pred_valid = evaluer(counts_valid, y_valid, assignations, 'Validation')
    res_test, pred_test = evaluer(counts_test, y_test, assignations, 'Test')

    duree = time.perf_counter() - debut
    print(f"Temps total : {duree / 60:.1f} min")

    # Figures
    print(f"Enregistrement des figures dans {dossier}")
    W = matrice_poids(reseau['input_synapse'])
    fig_champs_recepteurs(W, assignations, dossier)
    fig_histogramme_poids(W, dossier)
    fig_neurones_par_classe(assignations, dossier)
    fig_matrice_confusion(y_valid, pred_valid, dossier, 'validation')
    fig_matrice_confusion(y_test, pred_test, dossier, 'test')
    fig_activite(counts, n_train, n_valid, dossier)
    fig_courbes_accord(moyennes, assignations, dossier)
    fig_cartes_activation_classes(counts_valid, y_valid, dossier, 'validation')
    fig_exemples_activation(X_valid, counts_valid, y_valid, pred_valid, dossier, 'validation')
    if HOMEOSTASIE:
        fig_theta(reseau['excitatory_group'], dossier)

    enregistrer_resultats(res_valid, res_test, duree, n_train)


if __name__ == '__main__':
    main()
