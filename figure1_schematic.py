"""
Figure 1 - materials selection, control-group design and µCC loading scheme.

The figure is drawn entirely from geometric primitives in Matplotlib; no
photographs, stock images, generative-AI images or third-party artwork are
used. Running this script reproduces the figure exactly and writes the
software versions for the figure legend to figure1_legend.txt.

    python figure1_schematic.py            # writes to results/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import config

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
HEMP, BAMBOO, EPOXY, UCC = "#a8743c", "#d9b45a", "#e8d48a", "#ffffff"
PANEL = "#f6f6f4"


def _box(ax, x, y, w, h, fc="white", ec=INK2, lw=0.9, r=0.012, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _header(ax, x, y, text, color, w=0.34):
    _box(ax, x - w / 2, y - 0.022, w, 0.044, fc=color, ec=color, r=0.02, z=3)
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold", zorder=4)


def _arrow(ax, x, y0, y1):
    ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))


# ---------------------------------------------------------------- icons
def icon_hemp(ax, cx, cy, s=0.045):
    rng = np.random.default_rng(3)
    t = np.linspace(-1, 1, 60)
    for k in range(9):
        off = rng.uniform(-0.6, 0.6)
        ax.plot(cx + s * t, cy + s * (0.35 * np.sin(4 * t + k) * 0.4 + off * 0.5),
                color=HEMP, lw=1.3, solid_capstyle="round", zorder=3)


def icon_bamboo(ax, cx, cy, s=0.045):
    for k in range(5):
        y = cy - 0.6 * s + k * 0.3 * s
        ax.add_patch(Rectangle((cx - s, y), 2 * s, 0.17 * s, fc=BAMBOO, ec="#9c7a2b", lw=0.6, zorder=3))
        for nx in (-0.45, 0.2, 0.75):
            ax.plot([cx + nx * s] * 2, [y, y + 0.17 * s], color="#9c7a2b", lw=0.8, zorder=4)


def icon_beaker(ax, cx, cy, s=0.045, liquid=EPOXY):
    w, h = 0.75 * s, 1.25 * s
    ax.add_patch(Rectangle((cx - w, cy - h * 0.6), 2 * w, h * 0.85, fc=liquid, ec="none", zorder=2))
    ax.plot([cx - w, cx - w, cx + w, cx + w], [cy + h * 0.6, cy - h * 0.6, cy - h * 0.6, cy + h * 0.6],
            color=INK2, lw=1.2, zorder=3)


def icon_bottle(ax, cx, cy, s=0.045):
    ax.add_patch(FancyBboxPatch((cx - 0.5 * s, cy - 0.75 * s), s, 1.15 * s,
                                boxstyle="round,pad=0,rounding_size=0.006", fc="white", ec=INK2, lw=1.1, zorder=3))
    ax.add_patch(Rectangle((cx - 0.22 * s, cy + 0.4 * s), 0.44 * s, 0.3 * s, fc=ORANGE, ec=INK2, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((cx - 0.38 * s, cy - 0.35 * s), 0.76 * s, 0.4 * s, fc="#eaeaea", ec="none", zorder=4))


def icon_powder(ax, cx, cy, s=0.045):
    ax.add_patch(Polygon([[cx - 1.0 * s, cy - 0.15 * s], [cx + 1.0 * s, cy - 0.15 * s],
                          [cx + 0.75 * s, cy - 0.55 * s], [cx - 0.75 * s, cy - 0.55 * s]],
                         fc="#dfe6ee", ec=INK2, lw=1.0, zorder=3))
    rng = np.random.default_rng(1)
    for _ in range(70):
        x = rng.uniform(-0.8, 0.8)
        y = rng.uniform(-0.15, 0.25) * (1 - abs(x) * 0.8)
        ax.add_patch(Circle((cx + x * s, cy + y * s), 0.035 * s, fc=UCC, ec="#9aa4ad", lw=0.4, zorder=4))


def icon_dry(ax, cx, cy, s=0.045):
    for k, dx in enumerate((-0.45, 0.0, 0.45)):
        t = np.linspace(0, 1, 40)
        ax.plot(cx + dx * s + 0.12 * s * np.sin(10 * t), cy - 0.5 * s + t * s, color=ORANGE, lw=1.3, zorder=3)
    ax.text(cx, cy - 0.85 * s, "dried", ha="center", va="center", fontsize=7, color=INK2)


# ---------------------------------------------------------------- laminate
def laminate(ax, x, y, w=0.095, h=0.05, layers=("matrix",), dots=False, d=0.018):
    n = len(layers)
    lh = h / n
    col = {"matrix": "#eef1f4", "hemp": HEMP, "bamboo": BAMBOO}
    for i, kind in enumerate(layers):
        y0 = y + i * lh
        ax.add_patch(Rectangle((x, y0), w, lh, fc=col[kind], ec="white", lw=0.8, zorder=3))
        if kind == "hemp":
            t = np.linspace(0, 1, 40)
            for k in range(2):
                ax.plot(x + w * t, y0 + lh * (0.35 + 0.3 * k) + 0.1 * lh * np.sin(18 * t + k), color="#7a512a", lw=0.6, zorder=4)
        if kind == "bamboo":
            for k in range(2):
                ax.plot([x, x + w], [y0 + lh * (0.33 + 0.33 * k)] * 2, color="#9c7a2b", lw=0.6, zorder=4)
    top = [[x, y + h], [x + w, y + h], [x + w + d, y + h + d * 0.8], [x + d, y + h + d * 0.8]]
    side = [[x + w, y], [x + w + d, y + d * 0.8], [x + w + d, y + h + d * 0.8], [x + w, y + h]]
    ax.add_patch(Polygon(top, fc=col[layers[-1]], ec=INK2, lw=0.7, zorder=3, alpha=0.85))
    ax.add_patch(Polygon(side, fc=col[layers[-1]], ec=INK2, lw=0.7, zorder=3, alpha=0.65))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK2, lw=0.7, zorder=5))
    if dots:
        rng = np.random.default_rng(int(x * 1000))
        for _ in range(22):
            ax.add_patch(Circle((x + rng.uniform(0.05, 0.95) * w, y + rng.uniform(0.05, 0.95) * h),
                                0.0028, fc=UCC, ec="#6c7782", lw=0.4, zorder=6))


# ---------------------------------------------------------------- figure
def draw():
    fig = plt.figure(figsize=(7.2, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # panel 1 -----------------------------------------------------------
    _box(ax, 0.03, 0.735, 0.94, 0.225, fc=PANEL, ec=BLUE, lw=1.2, r=0.02)
    _header(ax, 0.5, 0.96, "1. Materials selection", BLUE)
    items = [("Hemp fibres", icon_hemp), ("Bamboo fibres", icon_bamboo), ("Epoxy resin", icon_beaker),
             ("Curing agent", icon_bottle), ("Microcrystalline\ncellulose (µCC)", icon_powder),
             ("Fibres and µCC\ndried before use", icon_dry)]
    for i, (lab, fn) in enumerate(items):
        cx = 0.105 + i * 0.158
        _box(ax, cx - 0.068, 0.755, 0.136, 0.17, fc="white", ec="#d6d5d0", r=0.012)
        fn(ax, cx, 0.86)
        ax.text(cx, 0.778, lab, ha="center", va="center", fontsize=7.6, color=INK)
    _arrow(ax, 0.5, 0.733, 0.703)

    # panel 2 -----------------------------------------------------------
    _box(ax, 0.03, 0.405, 0.94, 0.29, fc=PANEL, ec=VIOLET, lw=1.2, r=0.02)
    _header(ax, 0.5, 0.695, "2. Control groups (laminate types)", VIOLET, w=0.46)
    groups = [("Neat epoxy", ("matrix",) * 3, False),
              ("Hemp/epoxy", ("hemp",) * 4, False),
              ("Bamboo/epoxy", ("bamboo",) * 4, False),
              ("Hemp–bamboo/\nepoxy (unfilled)", ("hemp", "bamboo", "hemp", "bamboo"), False),
              ("Hemp–bamboo/\nepoxy + µCC\n(1, 2, 3, 5 wt%)", ("hemp", "bamboo", "hemp", "bamboo"), True),
              ("Hemp/epoxy\n+ µCC", ("hemp",) * 4, True),
              ("Bamboo/epoxy\n+ µCC", ("bamboo",) * 4, True)]
    for i, (lab, layers, dots) in enumerate(groups):
        cx = 0.1 + i * 0.133
        _box(ax, cx - 0.06, 0.475, 0.12, 0.185, fc="white", ec="#d6d5d0", r=0.012)
        laminate(ax, cx - 0.055, 0.575, w=0.09, h=0.048, layers=layers, dots=dots)
        ax.text(cx, 0.525, lab, ha="center", va="center", fontsize=7, color=INK, linespacing=1.15)
    _box(ax, 0.2, 0.418, 0.6, 0.04, fc="#ecebfa", ec="#ecebfa", r=0.015)
    ax.text(0.5, 0.438, "Each µCC-modified laminate compared with the unfilled hybrid laminate (0 wt%)",
            ha="center", va="center", fontsize=8, color=INK)
    _arrow(ax, 0.5, 0.403, 0.373)

    # panel 3 -----------------------------------------------------------
    _box(ax, 0.03, 0.03, 0.94, 0.335, fc=PANEL, ec=AQUA, lw=1.2, r=0.02)
    _header(ax, 0.5, 0.365, "3. Experimental design", AQUA)
    cols = [(0.05, "µCC loading levels"), (0.283, "Held constant"), (0.516, "Replication"), (0.749, "Responses")]
    for x, title in cols:
        _box(ax, x, 0.055, 0.2, 0.27, fc="white", ec="#d6d5d0", r=0.012)
        ax.text(x + 0.1, 0.3, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    shades = ["#e3eefb", "#b7d3f3", "#86b4ea", "#4f8fdd", "#1f5fae"]
    for i, (w, c) in enumerate(zip(config.UCC_LOADINGS_WT, shades)):
        x = 0.07 + i * 0.04
        ax.add_patch(Circle((x, 0.215), 0.016, fc=c, ec=INK2, lw=0.7, zorder=3))
        ax.text(x, 0.178, f"{w}", ha="center", va="center", fontsize=8, color=INK)
    ax.text(0.15, 0.15, "wt% of epoxy-resin mass", ha="center", va="center", fontsize=7.4, color=INK2)
    ax.text(0.15, 0.095, "0 wt% = unfilled\nhybrid baseline", ha="center", va="center", fontsize=7.4,
            color=INK2, linespacing=1.2)
    for i, t in enumerate(["Total fibre content", "Hemp : bamboo ratio", "Resin : hardener ratio",
                           "Laminate thickness", "Consolidation method", "Curing cycle"]):
        ax.text(0.293, 0.262 - i * 0.033, "•  " + t, ha="left", va="center", fontsize=7.6, color=INK)
    for k in range(3):
        laminate(ax, 0.576 + k * 0.012, 0.165 + k * 0.03, w=0.07, h=0.018, layers=("hemp", "bamboo"), d=0.012)
    ax.text(0.616, 0.12, "≥ 3 independent\nfabrication batches", ha="center", va="center", fontsize=7.6,
            color=INK, linespacing=1.2)
    ax.text(0.616, 0.075, "≥ 10 mechanical specimens\nper formulation", ha="center", va="center",
            fontsize=7, color=INK2, linespacing=1.2)
    for i, t in enumerate(["SEM image analysis", "Interface and fractography",
                           "Tensile, flexural, impact", "Interrupted-load damage",
                           "Water absorption, ageing", "TGA and DSC"]):
        ax.text(0.758, 0.262 - i * 0.033, "•  " + t, ha="left", va="center", fontsize=7.0, color=INK)
    return fig


def legend_text() -> str:
    return (
        "Figure 1. Materials selection, control-group design, and µCC loading scheme for "
        "hemp–bamboo/epoxy hybrid laminate fabrication. The schematic is original and was created "
        f"by the authors in Python {sys.version.split()[0]} (https://www.python.org) using Matplotlib "
        f"{matplotlib.__version__} (https://matplotlib.org); it contains no reproduced, modified or "
        "AI-generated images."
    )


def build(out=config.DEFAULT_OUTDIR):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw()
    fig.savefig(out / "figure1_schematic.png", dpi=config.FIG_DPI)
    fig.savefig(out / "figure1_schematic.pdf")
    fig.savefig(out / "figure1_schematic.svg")
    plt.close(fig)
    (out / "figure1_legend.txt").write_text(legend_text() + "\n", encoding="utf-8")
    return out / "figure1_schematic.png"


if __name__ == "__main__":
    p = build()
    print(legend_text())
    print(f"written: {p}")
