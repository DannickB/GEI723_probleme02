"""
GEI723 - Probleme 2, partie III : series of trainings, one parameter at a time.

Starting from the default parameters of diehl_cook_mnist.py, each run changes a
single parameter (no grid search). Every run saves its network in
resultats/serie/<name>/ and appends one line to resultats/serie/resultats.csv.
Runs already present in the CSV are skipped, so an interrupted series can
simply be restarted.

Both readouts are computed for every run from the saved spike counts, so the
readout comparison costs no extra training:
  - vote: Diehl & Cook (mean response of the neurons assigned to each class);
  - 1neurone: the previous readout (the single best neuron of each class).

Usage (from the project root):
    .venv\\Scripts\\python.exe PARTIE_2\\serie_entrainements.py

Estimated duration with 7000 images: about 8.5 h for the 11 runs below.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

import diehl_cook_mnist as dc

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_SERIE = os.path.join(DOSSIER, 'resultats', 'serie')
FICHIER_CSV = os.path.join(DOSSIER_SERIE, 'resultats.csv')

# (run name, parameters changed from the defaults of diehl_cook_mnist.py)
SERIE = [
    ('base', {}),
    ('epoques_2', {'N_EPOCHS': 2}),
    ('epoques_3', {'N_EPOCHS': 3}),
    ('neurones_100', {'N_NEURONS': 100}),
    ('neurones_400', {'N_NEURONS': 400}),
    ('homeostasie_off', {'theta_plus_e_mV': 0.0}),
    ('p_ie_0.5', {'P_IE': 0.5}),
    ('p_ei_0.5', {'P_EI': 0.5}),
    # The default nu_ee_post (0.01) is treated as the upper bound.
    ('nu_post_div3', {'nu_ee_post': 0.0033}),
    ('nu_post_div10', {'nu_ee_post': 0.001}),
    ('nu_post_div100', {'nu_ee_post': 0.0001}),
]

# Executed in a fresh Python process for each run: Brian2 standalone can only
# build one simulation per process. theta_plus_e needs a unit, hence the
# special case (JSON cannot carry Brian2 quantities).
CODE_RUN = """
import json, sys
import diehl_cook_mnist as dc
params = json.loads(sys.argv[1])
if 'theta_plus_e_mV' in params:
    dc.theta_plus_e = params.pop('theta_plus_e_mV') * dc.mV
for nom, valeur in params.items():
    setattr(dc, nom, valeur)
dc.main()
"""

COLONNES = ['nom', 'n_neurons', 'n_epochs', 'theta_plus_e_mV', 'p_ei', 'p_ie',
            'nu_ee_post', 'n_synapses',
            'acc_train_vote', 'acc_valid_vote', 'acc_test_vote',
            'acc_train_1neurone', 'acc_valid_1neurone', 'acc_test_1neurone',
            'duree_min', 'fichier']


def precision_1neurone(d):
    """Accuracies of the previous readout: the best neuron of each class.

    The best neuron is chosen on the frozen labeling pass, like the vote readout.

    Args:
        d: Contents of a .npz written by diehl_cook_mnist.sauvegarder_reseau().

    Returns:
        Tuple (train, valid, test) of accuracies.
    """
    counts, y = d['counts_etiquetage'].astype(float), d['y_train']
    reponse = np.array([counts[y == c].mean(axis=0) for c in range(10)])
    meilleurs = reponse.argmax(axis=1)
    return tuple(np.mean(c[:, meilleurs].argmax(axis=1) == yy)
                 for c, yy in [(d['counts_etiquetage'], d['y_train']),
                               (d['counts_valid'], d['y_valid']),
                               (d['counts_test'], d['y_test'])])


def runs_faits():
    if not os.path.exists(FICHIER_CSV):
        return set()
    with open(FICHIER_CSV, newline='', encoding='utf-8') as f:
        return {ligne['nom'] for ligne in csv.DictReader(f)}


def ajouter_ligne(ligne):
    nouveau = not os.path.exists(FICHIER_CSV)
    with open(FICHIER_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES)
        if nouveau:
            writer.writeheader()
        writer.writerow(ligne)


def lancer(nom, params):
    """Train one configuration in a subprocess and record its results.

    Args:
        nom: Run name, also used as its results folder.
        params: Parameters overriding the defaults of diehl_cook_mnist.py.
    """
    dossier_run = os.path.join(DOSSIER_SERIE, nom)
    params_run = dict(params, DOSSIER_RESULTATS=dossier_run)
    # Agg backend: plt.show() at the end of main() must not block the series.
    env = dict(os.environ, MPLBACKEND='Agg')

    print(f"\n===== {nom} : {params or 'valeurs par defaut'} =====", flush=True)
    debut = time.time()
    subprocess.run([sys.executable, '-c', CODE_RUN, json.dumps(params_run)],
                   cwd=DOSSIER, env=env, check=True)
    duree = (time.time() - debut) / 60

    fichier = max(glob.glob(os.path.join(dossier_run, '*.npz')), key=os.path.getmtime)
    d = np.load(fichier)
    acc_1n = precision_1neurone(d)
    ajouter_ligne({
        'nom': nom,
        'n_neurons': params.get('N_NEURONS', dc.N_NEURONS),
        'n_epochs': params.get('N_EPOCHS', dc.N_EPOCHS),
        'theta_plus_e_mV': params.get('theta_plus_e_mV', float(dc.theta_plus_e / dc.mV)),
        'p_ei': params.get('P_EI', dc.P_EI),
        'p_ie': params.get('P_IE', dc.P_IE),
        'nu_ee_post': params.get('nu_ee_post', dc.nu_ee_post),
        'n_synapses': len(d['entree_i']) + len(d['ei_i']) + len(d['ie_i']),
        'acc_train_vote': round(float(d['acc_train']), 4),
        'acc_valid_vote': round(float(d['acc_valid']), 4),
        'acc_test_vote': round(float(d['acc_test']), 4),
        'acc_train_1neurone': round(acc_1n[0], 4),
        'acc_valid_1neurone': round(acc_1n[1], 4),
        'acc_test_1neurone': round(acc_1n[2], 4),
        'duree_min': round(duree, 1),
        'fichier': os.path.relpath(fichier, DOSSIER),
    })


def main():
    os.makedirs(DOSSIER_SERIE, exist_ok=True)
    faits = runs_faits()
    for nom, params in SERIE:
        if nom in faits:
            print(f"{nom} : deja dans le CSV, ignore")
            continue
        try:
            lancer(nom, params)
        except subprocess.CalledProcessError:
            # Keep going: one failed configuration should not stop a night-long series.
            print(f"{nom} : ECHEC, run suivant")
    print(f"\nResultats : {FICHIER_CSV}")


if __name__ == '__main__':
    main()
