"""Schéma de la validation croisée imbriquée (figure du rapport).

Lancer depuis la racine du projet :  python3 report/schema_validation_imbriquee.py
Écrit report/figures/validation_imbriquee.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ENCRE, ENCRE_2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"
BLEU, ORANGE = "#2a78d6", "#eb6834"
BLEU_CLAIR, ORANGE_CLAIR = "#a9c9f0", "#f5b496"
plt.rcParams.update({"font.size": 8, "text.color": ENCRE})

fig, ax = plt.subplots(figsize=(6.9, 3.5))
ax.set_xlim(0, 14.2)
ax.set_ylim(-1.35, 6.6)
ax.axis("off")


def bloc(x, y, w, h, couleur, texte="", couleur_texte="white", taille=7):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=couleur, edgecolor="white", linewidth=1.2))
    if texte:
        ax.text(x + w / 2, y + h / 2, texte, ha="center", va="center", color=couleur_texte, fontsize=taille)


def fleche(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                 color=ENCRE_2, linewidth=0.9, linestyle=ls))


# --- Boucle extérieure : 5 tours, test = un groupe d'articles à tour de rôle
ax.text(0, 6.35, "Boucle extérieure : mesurer le score", fontsize=9, fontweight="bold")
ax.text(0, 5.95, "articles répartis en 5 groupes A à E", color=ENCRE_2, fontsize=7)
groupes = "ABCDE"
W, H, X0 = 0.78, 0.62, 0.9
for tour in range(5):
    y = 4.9 - tour * 0.9
    test = 4 - tour  # tour 1 : E, tour 2 : D, ...
    ax.text(0, y + H / 2, f"Tour {tour + 1}", va="center", fontsize=7.5, color=ENCRE_2)
    for k, g in enumerate(groupes):
        bloc(X0 + k * W, y, W, H, ORANGE if k == test else BLEU, g)
    ax.text(X0 + 5 * W + 0.15, y + H / 2, f"→ score {tour + 1}", va="center", fontsize=7.5)
ax.text(X0, -0.15, "Score final = moyenne des 5 scores", fontsize=8, fontweight="bold")
# légende
bloc(X0, -1.0, 0.32, 0.3, BLEU)
ax.text(X0 + 0.4, -0.85, "entraînement", va="center", fontsize=7)
bloc(X0 + 2.3, -1.0, 0.32, 0.3, ORANGE)
ax.text(X0 + 2.7, -0.85, "test", va="center", fontsize=7)
bloc(X0 + 3.5, -1.0, 0.32, 0.3, ORANGE_CLAIR)
ax.text(X0 + 3.9, -0.85, "validation (choix du réglage)", va="center", fontsize=7)

XI = 7.4
# --- Lien tour 1 (partie entraînement A-D) -> boucle intérieure
y1 = 4.9
ax.plot([X0, X0 + 4 * W], [y1 + H + 0.1] * 2, color=ENCRE_2, linewidth=0.9)
for x in (X0, X0 + 4 * W):
    ax.plot([x, x], [y1 + H + 0.02, y1 + H + 0.1], color=ENCRE_2, linewidth=0.9)
ax.plot([X0 + 2 * W] * 2, [y1 + H + 0.1, y1 + H + 0.22], color=ENCRE_2, linewidth=0.9, linestyle="--")
fleche(X0 + 2 * W, y1 + H + 0.22, XI - 0.1, y1 + H + 0.22, style="-|>", ls="--")

# --- Boucle intérieure : la partie entraînement redécoupée en 5, validation à tour de rôle
ax.text(XI, 6.35, "Boucle intérieure : choisir le réglage", fontsize=9, fontweight="bold")
ax.text(XI, 5.95, "dans le tour 1 : A, B, C, D seulement, redécoupés en 5 parts", color=ENCRE_2, fontsize=7)
WI, HI = 0.62, 0.44
for r in range(5):
    y = 5.0 - r * 0.6
    for k in range(5):
        bloc(XI + k * WI, y, WI, HI, ORANGE_CLAIR if k == 4 - r else BLEU_CLAIR)
ax.text(XI + 5 * WI + 0.2, 4.0,
        "pour chaque réglage\n(profondeur 2, 3, 4…) :\nmoyenne des 5 scores\nde validation",
        va="center", fontsize=7)
fleche(XI + 2.5 * WI, 2.5, XI + 2.5 * WI, 1.95)
ax.text(XI + 2.5 * WI, 1.7, "meilleur réglage", ha="center", va="center", fontsize=8, fontweight="bold")
fleche(XI + 2.5 * WI, 1.45, XI + 2.5 * WI, 0.9)
ax.text(XI + 2.5 * WI, 0.55, "entraînement sur A, B, C, D avec ce réglage,\npuis score sur E (jamais vu) → score 1",
        ha="center", va="center", fontsize=7.5)

fig.savefig(Path("report/figures/validation_imbriquee.pdf"), bbox_inches="tight")
