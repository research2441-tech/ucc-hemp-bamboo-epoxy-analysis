"""
Publication figures generated directly from the analysed data.

  Fig. 3  microstructural characteristics vs µCC loading (Table 2)
  Fig. 4  interfacial and damage indicators vs µCC loading (Table 3)
  Fig. 6  Tukey HSD post-hoc comparison of tensile strength:
          (a) mean ± SD with compact letter display, (b) all pairwise
          differences with Tukey family-wise 95 % confidence intervals
  Fig. S1 tensile, flexural and impact strength with Tukey letters
  Fig. S2 crack-density evolution under interrupted loading (Table 7)
  Fig. S3 TGA / DSC characteristic temperatures (Table 11)

Letters above bars come from the same Tukey HSD computation written to
results/tukey_pairwise.csv, so the figure and the text cannot disagree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

C = config.COLORS
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.edgecolor": C["ink2"], "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.labelcolor": C["ink"], "axes.grid": True,
    "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

LOAD_COLORS = {"0": C["series1"], "2": C["series2"], "3": C["series3"], "5": "#eda100"}


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=config.FIG_DPI)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def bar_panel(ax, wts, means, sds, letters=None, ylabel="", title="", color=C["series1"]):
    x = np.arange(len(wts))
    ok = ~np.isnan(means)
    ax.bar(x[ok], np.asarray(means)[ok], width=0.62, color=color, edgecolor="white", linewidth=0.6, zorder=2)
    ax.errorbar(x[ok], np.asarray(means)[ok], yerr=np.asarray(sds)[ok], fmt="none",
                ecolor=C["ink"], elinewidth=1, capsize=3, zorder=3)
    top = np.nanmax(np.asarray(means) + np.asarray(sds))
    if letters:
        for xi, w, m, s in zip(x, wts, means, sds):
            if not np.isnan(m):
                ax.text(xi, m + s + 0.03 * top, letters.get(str(w), ""), ha="center",
                        va="bottom", fontsize=8.5, color=C["ink"], fontweight="bold")
    ax.set_xticks(x, [str(w) for w in wts])
    ax.set_xlabel("µCC loading (wt%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.16)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)


def panel_grid(df_summary, specs, letters, out, name, ncols=3):
    n = len(specs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i, (resp, level, ylabel) in enumerate(specs):
        sub = df_summary[(df_summary.response == resp) & (df_summary.load_level == level) & (df_summary.table != "T10")]
        sub = sub.sort_values("ucc_wt", key=lambda s: s.astype(float))
        bar_panel(axes[i], sub.ucc_wt.tolist(), sub["mean"].to_numpy(float), sub["sd"].to_numpy(float),
                  letters.get((resp, level)), ylabel, f"({chr(97 + i)})")
    for ax in axes[n:]:
        ax.set_visible(False)
    _save(fig, out, name)


def fig_tukey_tensile(df_summary, pairwise, letters, out):
    sub = df_summary[(df_summary.response == "tensile_strength") & (df_summary.load_level == "") & (df_summary.table != "T10")]
    sub = sub.sort_values("ucc_wt", key=lambda s: s.astype(float))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.35]})
    bar_panel(a1, sub.ucc_wt.tolist(), sub["mean"].to_numpy(float), sub["sd"].to_numpy(float),
              letters, "Tensile strength (MPa)", "(a)")
    pw = pairwise[(pairwise.response == "tensile_strength")].reset_index(drop=True)
    y = np.arange(len(pw))[::-1]
    for yi, (_, r) in zip(y, pw.iterrows()):
        col = C["series1"] if r.significant else C["ink2"]
        a2.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=2, solid_capstyle="round")
        a2.plot(r.mean_diff, yi, "o", ms=6, mfc=col if r.significant else "white", mec=col, mew=1.5)
        ptxt = "p < 0.001" if r.p_adj < 0.001 else f"p = {r.p_adj:.3f}"
        a2.annotate(ptxt, (max(r.ci_high, 0) + 0.6, yi), va="center", fontsize=7, color=C["ink2"])
    a2.axvline(0, color=C["ink"], lw=0.8, ls="--")
    a2.set_yticks(y, [f"{r.group_b} − {r.group_a} wt%" for _, r in pw.iterrows()])
    a2.set_xlabel("Difference in mean tensile strength (MPa)\nTukey HSD family-wise 95 % CI")
    a2.set_title("(b)", loc="left")
    a2.grid(axis="y", visible=False)
    lo, hi = pw.ci_low.min(), pw.ci_high.max()
    a2.set_xlim(min(lo, 0) - 1.5, hi + 8)
    a2.plot([], [], "o", mfc=C["series1"], mec=C["series1"], label="significant (p < 0.05)")
    a2.plot([], [], "o", mfc="white", mec=C["ink2"], label="not significant")
    a2.legend(loc="lower right", fontsize=7)
    _save(fig, out, "fig6_tukey_tensile_strength")


def fig_crack_evolution(df_summary, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    levels = ["25", "50", "75", "90"]
    for w, col in LOAD_COLORS.items():
        sub = df_summary[(df_summary.response == "crack_density") & (df_summary.ucc_wt == w)]
        sub = sub.set_index("load_level").loc[levels]
        x = np.array(levels, float)
        ax.errorbar(x, sub["mean"], yerr=sub["sd"], color=col, lw=2, marker="o", ms=5,
                    capsize=2.5, label=f"{w} wt%")
        ax.annotate(f"{w} wt%", (x[-1] + 1.2, sub["mean"].iloc[-1]), va="center", fontsize=7.5, color=C["ink"])
    ax.set_xlim(20, 99)
    ax.set_xlabel("Load level (% of ultimate load)")
    ax.set_ylabel("Crack density (mm$^{-1}$)")
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, out, "figS2_crack_density_evolution")


def make_all(df_summary, pairwise, letters, out: Path):
    out = Path(out)
    panel_grid(df_summary, [("agglomerate_size", "", "Mean agglomerate size (µm)"),
                            ("agglomerate_area_fraction", "", "Agglomerate area fraction (%)"),
                            ("agglomerate_number_density", "", "Number density (mm$^{-2}$)"),
                            ("void_fraction", "", "Void fraction (%)"),
                            ("resin_rich_area", "", "Resin-rich area (%)"),
                            ("fiber_spacing_variation", "", "Fibre-spacing variation (%)")],
               letters, out, "fig3_microstructure")
    panel_grid(df_summary, [("interfacial_gap_index", "", "Interfacial-gap index (–)"),
                            ("debonded_interface_frequency", "", "Debonded interface (%)"),
                            ("pullout_frequency", "", "Fibre pull-out frequency (%)"),
                            ("mean_pullout_length", "", "Mean pull-out length (µm)"),
                            ("ucc_matrix_separation", "", "µCC–matrix separation (%)")],
               letters, out, "fig4_interface")
    panel_grid(df_summary, [("tensile_strength", "", "Tensile strength (MPa)"),
                            ("flexural_strength", "", "Flexural strength (MPa)"),
                            ("impact_strength", "", "Impact strength (kJ m$^{-2}$)")],
               letters, out, "figS1_mechanical_tukey")
    panel_grid(df_summary, [("degradation_onset", "", "Degradation onset (°C)"),
                            ("max_degradation_temperature", "", "T$_{max}$ (°C)"),
                            ("glass_transition_temperature", "", "T$_g$ (°C)")],
               letters, out, "figS3_thermal")
    fig_tukey_tensile(df_summary, pairwise, letters.get(("tensile_strength", ""), {}), out)
    fig_crack_evolution(df_summary, out)
