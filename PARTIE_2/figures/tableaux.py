import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm, colors

CMAP = matplotlib.colormaps['Blues']
NORM = colors.Normalize(0.20, 0.85)

def couleur(acc):
    # Stay in the pale half of Blues so black text remains readable.
    return CMAP(0.05 + 0.55 * NORM(acc))

def tableau(lignes, entetes, titre, note, fichier, largeurs=(3.0, 2.6, 2.4)):
    """Draw a poster table and save it as a PNG.

    Args:
        lignes: Rows (parameter, value, accuracy, ...). Each accuracy column
            gets a blue background; None leaves the cell empty ("—").
        entetes: Column headers.
        titre: Title above the table.
        note: Small italic line below the table.
        fichier: Output PNG path.
        largeurs: Column widths in inches.
    """
    h, n = 0.55, len(lignes)
    W, H = sum(largeurs), h * (n + 1)
    fig, ax = plt.subplots(figsize=(W, H + 0.9), dpi=300)
    ax.set_xlim(0, W); ax.set_ylim(-0.55, H + 0.35); ax.axis('off')
    ax.text(0, H + 0.15, titre, fontsize=13, weight='bold', va='bottom')
    xs = [sum(largeurs[:i]) for i in range(len(largeurs))]
    y = H - h
    ax.add_patch(Rectangle((0, y), W, h, color='#1f3b57'))
    for c, (x, w, t) in enumerate(zip(xs, largeurs, entetes)):
        ax.text(x + 0.15 if c < 2 else x + w / 2, y + h / 2, t, color='white',
                fontsize=11, weight='bold', va='center', ha='left' if c < 2 else 'center')
    meilleurs = [max(l[c] for l in lignes if l[c] is not None) for c in range(2, len(largeurs))]
    precedent = None
    for k, (param, valeur, *accs) in enumerate(lignes):
        y = H - h * (k + 2)
        ax.add_patch(Rectangle((0, y), W, h, color='#f6f7f9' if k % 2 else 'white'))
        if param != precedent:
            ax.text(0.15, y + h / 2, param, fontsize=11, va='center', weight='bold')
            if k:
                ax.plot([0, W], [y + h, y + h], color='#c9ced6', lw=0.8)
        precedent = param
        ax.text(xs[1] + 0.15, y + h / 2, valeur, fontsize=11, va='center')
        for x, w, acc, best in zip(xs[2:], largeurs[2:], accs, meilleurs):
            texte = '—'
            if acc is not None:
                ax.add_patch(Rectangle((x, y), w, h, color=couleur(acc)))
                texte = f"{acc*100:.1f} %".replace('.', ',')
            ax.text(x + w / 2, y + h / 2, texte, fontsize=11, va='center', ha='center',
                    weight='bold' if acc == best else 'normal')
    ax.plot([0, W], [0, 0], color='#1f3b57', lw=1.2)
    ax.text(0, -0.3, note, fontsize=8.5, color='#555', va='center', style='italic')
    fig.savefig(fichier, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    print(fichier)

ENT = ['Paramètre', 'Valeur testée', 'Précision (valid.)']
NOTE = "7 000 images, 200 neurones · référence : 1 époque, P_EI = P_IE = 1, ν_post = 0,01"

tableau([
    ('Référence', '—', 0.566),
    ('Itérations', '2 époques', 0.793),
    ('Itérations', '3 époques', 0.757),
    ('Neurones', '100', 0.687),
    ('Neurones', '400', 0.376),
    ('Homéostasie', 'désactivée', 0.266),
    ('Connexion E→I', 'P = 0,5', 0.773),
    ('Connexion I→E', 'P = 0,5', 0.494),
    ('Taux ν_post', '0,0033', 0.569),
    ('Taux ν_post', '0,001', 0.767),
    ('Taux ν_post', '0,0001', 0.427),
    ('Lecture', '1 neurone / classe', 0.214),
], ENT, 'Exploration des paramètres', NOTE, 'figures/tableau_exploration.png')

tableau([
    ('Référence', '—', 0.566),
    ('ν_post + itérations', '0,001 ; 2 époques', 0.794),
    ('ν_post + E→I', '0,001 ; P = 0,5', 0.737),
    ('ν_post + itérations + E→I', '0,001 ; 2 ép. ; P = 0,5', 0.697),
], ['Paramètres combinés', 'Valeurs testées', 'Précision (valid.)'], 'Combinaisons de paramètres', NOTE,
   'figures/tableau_combinaisons.png', largeurs=(3.6, 3.0, 2.4))

# The test set is reported once, for the configuration chosen on validation.
tableau([
    ('Homéostasie θ+', '0,05 mV', 0.791, None),
    ('Homéostasie θ+', '0,025 mV', 0.828, 0.835),
], ['Paramètre', 'Valeur testée', 'Précision (valid.)', 'Précision (test)'],
   'Entraînement complet', "70 000 images, 200 neurones, 2 époques",
   'figures/tableau_entrainement_complet.png', largeurs=(3.0, 2.2, 2.4, 2.2))
