"""
One-command reproduction of the statistical results, tables and figures.

    python run_all.py                    # summary statistics of Tables 2-12
    python run_all.py --specimens specimen_data.csv
    python run_all.py --out results --skip-selftests

Outputs (written to results/):
  table9_anova.csv / .md         one-way ANOVA: F(df1, df2), p, partial η², ω²,
                                 Welch F and Bartlett p for every Table 9 response
  anova_all_responses.csv        the same for every tabulated response
  tukey_pairwise.csv             Tukey HSD (Tukey-Kramer) pairwise comparisons
  games_howell_pairwise.csv      unequal-variance post-hoc comparisons
  tukey_letters.csv              compact letter display used in Figs 3, 4, 6
  descriptives.csv               mean, SD, n, CV, 95 % CI, % change vs 0 wt%
  control_groups.csv             Table 10: hybridization and µCC contributions
  loading_regimes.csv            data-driven regime assignment (Table 13)
  correlations.csv               Pearson / Spearman relationships (Section 4.7)
  regression_tensile.csv         standardised regression and LMG shares (Table 8)*
  consistency_audit.md / .csv    manuscript numerical consistency audit
  fig*.png / fig*.pdf            figures, 600 dpi raster and vector
  figure1_*.png/.pdf/.svg        Figure 1 schematic and its legend text
  summary.md                     one-page report of all of the above

  * requires specimen-level data (--specimens)
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import config
import consistency_audit as audit
import correlation_regression as cr
import figures
import stats_core as sc


def _md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _groups_summary(df, response, level):
    sub = df[(df.response == response) & (df.load_level == level) & (df.table != "T10")]
    sub = sub[sub["mean"].notna()].copy()
    sub = sub.sort_values("ucc_wt", key=lambda s: s.astype(float))
    return sub.ucc_wt.tolist(), sub["mean"].to_numpy(float), sub["sd"].to_numpy(float), sub["n"].to_numpy(int)


def _raw_column(response, level):
    return f"{response}_{level}" if level else response


def analyse_response(df, response, level, raw=None):
    """ANOVA, assumption checks and post-hoc tests for one response."""
    col = _raw_column(response, level)
    source = "summary statistics"
    checks = {}
    if raw is not None and col in raw and raw[col].notna().sum() > 0:
        grp = {str(int(w)): g[col].dropna().to_numpy() for w, g in raw.groupby("ucc_wt")}
        grp = {k: v for k, v in grp.items() if v.size >= 2}
        labels = sorted(grp, key=float)
        means = np.array([grp[k].mean() for k in labels])
        sds = np.array([grp[k].std(ddof=1) for k in labels])
        ns = np.array([grp[k].size for k in labels])
        checks = sc.assumption_checks_raw([grp[k] for k in labels])
        source = "specimen data"
    else:
        labels, means, sds, ns = _groups_summary(df, response, level)
    if len(labels) < 2:
        return None
    an = sc.anova_from_summary(means, sds, ns)
    welch = sc.welch_anova_from_summary(means, sds, ns)
    bart = sc.bartlett_from_summary(sds, ns)
    tk = sc.tukey_from_summary(labels, means, sds, ns, config.ALPHA)
    gh = sc.games_howell_from_summary(labels, means, sds, ns, config.ALPHA)
    letters = sc.compact_letter_display(labels, means, tk)
    row = {"response": response, "load_level": level, "source": source, "groups": len(labels),
           "n_per_group": "/".join(map(str, ns)), **an.as_dict(),
           "report": an.apa(), "welch_F": welch["F"], "welch_df2": welch["df_within"],
           "welch_p": welch["p"], "bartlett_chi2": bart["statistic"], "bartlett_p": bart["p"],
           **{k: v for k, v in checks.items() if not isinstance(v, list)}}
    for r in tk:
        r.update(response=response, load_level=level)
    for r in gh:
        r.update(response=response, load_level=level)
    return row, tk, gh, letters, (labels, means, sds, ns)


def control_groups(df):
    rows = []
    t10 = df[df.table == "T10"]
    for resp in config.MECHANICAL_RESPONSES:
        sub = t10[t10.response == resp].set_index("system").loc[config.CONTROL_SYSTEMS]
        m, s, n = sub["mean"].to_numpy(float), sub["sd"].to_numpy(float), sub["n"].to_numpy(int)
        an = sc.anova_from_summary(m, s, n)
        tk = {(r["group_a"], r["group_b"]): r for r in
              sc.tukey_from_summary(list(sub.index), m, s, n, config.ALPHA)}
        hemp, bam, hyb = sub.loc["hemp/epoxy", "mean"], sub.loc["bamboo/epoxy", "mean"], sub.loc["hemp-bamboo/epoxy", "mean"]
        f = config.HEMP_MASS_FRACTION_IN_HYBRID
        rom = f * hemp + (1 - f) * bam
        for sys_, r in sub.iterrows():
            key = (sys_, "hemp-bamboo/epoxy") if (sys_, "hemp-bamboo/epoxy") in tk else ("hemp-bamboo/epoxy", sys_)
            p = tk[key]["p_adj"] if key in tk else np.nan
            rows.append({"response": resp, "system": sys_, "mean": r["mean"], "sd": r["sd"],
                         "change_vs_neat_epoxy_pct": sc.percent_change(r["mean"], sub.loc["neat epoxy", "mean"]),
                         "change_vs_unfilled_hybrid_pct": sc.percent_change(r["mean"], hyb),
                         "tukey_p_vs_unfilled_hybrid": p,
                         "anova": an.apa()})
        rows.append({"response": resp, "system": "hybrid effect vs rule of mixtures",
                     "mean": hyb, "sd": np.nan, "change_vs_neat_epoxy_pct": np.nan,
                     "change_vs_unfilled_hybrid_pct": sc.percent_change(hyb, rom),
                     "tukey_p_vs_unfilled_hybrid": np.nan,
                     "anova": f"ROM ({f:.2f} hemp) = {rom:.1f}"})
    return pd.DataFrame(rows)


def loading_regimes(letters, pairwise):
    """
    Effective reinforcement : shares a Tukey letter with the best formulation in
                              the majority of tensile, flexural and impact strength
    Excessive loading       : above the optimum and at least one defect metric
                              (void fraction, gap index, agglomerate area fraction)
                              significantly higher than the unfilled laminate
    Low loading             : all remaining formulations
    """
    rows = []
    best = {}
    for r in config.MECHANICAL_RESPONSES:
        L = letters[(r, "")]
        best[r] = [k for k, v in L.items() if "a" in v]
    opt = max(config.UCC_LOADINGS_WT, key=lambda w: sum(str(w) in best[r] for r in config.MECHANICAL_RESPONSES))
    for w in config.UCC_LOADINGS_WT:
        w = str(w)
        share = sum(bool(set(letters[(r, "")][w]) & set("a")) for r in config.MECHANICAL_RESPONSES)
        defect_up = []
        for d in config.DEFECT_RESPONSES:
            pw = pairwise[(pairwise.response == d) & (pairwise.load_level == "")]
            hit = pw[((pw.group_a == "0") & (pw.group_b == w)) & pw.significant & (pw.mean_diff > 0)]
            if len(hit):
                defect_up.append(d)
        if share >= 2:
            regime = "effective reinforcement"
        elif float(w) > float(opt) and defect_up:
            regime = "excessive loading"
        else:
            regime = "low loading"
        rows.append({"ucc_wt": w, "regime": regime, "mechanical_responses_in_top_group": share,
                     "defect_metrics_significantly_above_unfilled": ", ".join(defect_up) or "none"})
    return pd.DataFrame(rows)


def correlations(df, raw):
    if raw is not None:
        d = cr.add_uniformity(raw)
        tab = cr.correlation_table(d)
        tab["level"] = "specimen"
        reg = model = None
        try:
            reg, model = cr.standardized_regression(d)
        except (ValueError, KeyError) as e:
            print(f"  regression skipped: {e}")
        return tab, reg, model
    wide = (df[(df.table != "T10") & (df.load_level == "")]
            .pivot_table(index="ucc_wt", columns="response", values="mean"))
    wide = cr.add_uniformity(wide.reset_index())
    tab = cr.correlation_table(wide)
    tab["level"] = "formulation means (n = 5, indicative)"
    return tab, None, None


def run_selftests():
    import image_analysis, moisture_retention, thermal_analysis
    res = {"image_analysis": image_analysis.selftest(False),
           "thermal_analysis": thermal_analysis.selftest(False),
           "moisture_retention": moisture_retention.selftest(False),
           "correlation_regression": cr.selftest(False)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.REPORTED_DATA))
    ap.add_argument("--specimens", default=None, help="specimen-level CSV (see specimen_data_template.csv)")
    ap.add_argument("--out", default=str(config.DEFAULT_OUTDIR))
    ap.add_argument("--skip-selftests", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = audit.load(a.data)
    spec_path = Path(a.specimens) if a.specimens else (config.SPECIMEN_DATA if config.SPECIMEN_DATA.exists() else None)
    raw = pd.read_csv(spec_path) if spec_path else None
    if raw is not None and raw.empty:
        raw = None
    print(f"Data: {a.data}" + (f" + specimen data {spec_path}" if raw is not None else ""))

    # ---------------- ANOVA + post-hoc for every response -----------------
    keys = (df[df.table != "T10"][["response", "load_level"]].drop_duplicates().itertuples(index=False))
    anova_rows, tk_rows, gh_rows, letters = [], [], [], {}
    for resp, lev in keys:
        res = analyse_response(df, resp, lev, raw)
        if res is None:
            continue
        row, tk, gh, L, _ = res
        anova_rows.append(row)
        tk_rows += tk
        gh_rows += gh
        letters[(resp, lev)] = L
    anova = pd.DataFrame(anova_rows)
    pairwise = pd.DataFrame(tk_rows)
    anova.to_csv(out / "anova_all_responses.csv", index=False)
    pairwise.to_csv(out / "tukey_pairwise.csv", index=False)
    pd.DataFrame(gh_rows).to_csv(out / "games_howell_pairwise.csv", index=False)
    pd.DataFrame([{"response": r, "load_level": l, "ucc_wt": w, "letters": v}
                  for (r, l), L in letters.items() for w, v in L.items()]).to_csv(out / "tukey_letters.csv", index=False)

    # ---------------- Table 9 --------------------------------------------
    t9 = []
    for resp, lev, label in config.TABLE9_RESPONSES:
        r = anova[(anova.response == resp) & (anova.load_level == lev)].iloc[0]
        t9.append({"Response": label, "F": round(r.F, 2), "df_between": int(r.df_between),
                   "df_within": int(r.df_within), "p": r.p, "partial_eta_sq": round(r.partial_eta_sq, 2),
                   "omega_sq": round(r.omega_sq, 2), "welch_F": round(r.welch_F, 2),
                   "welch_p": r.welch_p, "bartlett_p": round(r.bartlett_p, 3), "report": r.report})
    t9 = pd.DataFrame(t9)
    t9.to_csv(out / "table9_anova.csv", index=False)
    with open(out / "table9_anova.md", "w", encoding="utf-8") as f:
        f.write("| Response | F (df1, df2) | p-value | Partial η² | ω² | Welch F (p) | Bartlett p |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in t9.iterrows():
            p = "<0.001" if r.p < 0.001 else f"{r.p:.3f}"
            wp = "<0.001" if r.welch_p < 0.001 else f"{r.welch_p:.3f}"
            f.write(f"| {r.Response} | {r.F:.2f} ({r.df_between}, {r.df_within}) | {p} | "
                    f"{r.partial_eta_sq:.2f} | {r.omega_sq:.2f} | {r.welch_F:.2f} ({wp}) | {r.bartlett_p:.3f} |\n")

    # ---------------- descriptives -----------------------------------------
    d = df[(df.table != "T10") & df["mean"].notna()].copy()
    lo, hi = zip(*[sc.ci_from_summary(m, s, n, config.CI_LEVEL) for m, s, n in zip(d["mean"], d["sd"], d["n"])])
    d["cv_pct"], d["ci95_low"], d["ci95_high"] = 100 * d["sd"] / d["mean"], lo, hi
    base = d[d.ucc_wt == "0"].set_index(["response", "load_level"])["mean"]
    d["change_vs_0wt_pct"] = [sc.percent_change(m, base.get((r, l), np.nan)) for m, r, l in
                              zip(d["mean"], d.response, d.load_level)]
    d.to_csv(out / "descriptives.csv", index=False)

    # ---------------- control groups, regimes, correlations ---------------
    cg = control_groups(df)
    cg.to_csv(out / "control_groups.csv", index=False)
    reg_df = loading_regimes(letters, pairwise)
    reg_df.to_csv(out / "loading_regimes.csv", index=False)
    corr, reg, model = correlations(df, raw)
    corr.to_csv(out / "correlations.csv", index=False)
    if reg is not None:
        reg.to_csv(out / "regression_tensile.csv", index=False)

    # ---------------- audit, figures, self-tests --------------------------
    aud = audit.run(df, out)
    if not a.skip_figures:
        figures.make_all(df, pairwise, letters, out)
        import figure1_schematic
        figure1_schematic.build(out)
    st = {} if a.skip_selftests else run_selftests()

    # ---------------- summary report --------------------------------------
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Reproduction summary\n\n")
        f.write(f"Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__} | "
                f"pandas {pd.__version__}\n\n")
        f.write(f"Data source: {'specimen-level data' if raw is not None else 'summary statistics (mean, SD, n) of Tables 2-12'}\n\n")
        f.write("## Table 9 - one-way ANOVA (factor: µCC loading)\n\n")
        f.write(open(out / "table9_anova.md", encoding="utf-8").read() + "\n")
        f.write("## Tukey HSD letters (groups sharing a letter do not differ, α = 0.05)\n\n")
        f.write("| Response | " + " | ".join(f"{w} wt%" for w in config.UCC_LOADINGS_WT) + " |\n")
        f.write("|---|" + "---|" * len(config.UCC_LOADINGS_WT) + "\n")
        for (r, l), L in letters.items():
            f.write(f"| {r}{' ' + l + '%' if l else ''} | " +
                    " | ".join(L.get(str(w), "–") for w in config.UCC_LOADINGS_WT) + " |\n")
        f.write("\n## Tensile strength - Tukey HSD pairwise comparisons\n\n")
        f.write("| Comparison | Δ mean (MPa) | 95 % CI | p (adj.) | Significant |\n|---|---|---|---|---|\n")
        for _, r in pairwise[pairwise.response == "tensile_strength"].iterrows():
            p = "<0.001" if r.p_adj < 0.001 else f"{r.p_adj:.3f}"
            f.write(f"| {r.group_b} vs {r.group_a} wt% | {r.mean_diff:+.1f} | {r.ci_low:.1f} to {r.ci_high:.1f} | "
                    f"{p} | {'yes' if r.significant else 'no'} |\n")
        f.write("\n## Loading regimes\n\n" + _md(reg_df) + "\n\n")
        f.write(f"## Consistency audit\n\n{int(aud['pass'].sum())}/{len(aud)} checks passed "
                f"(details in consistency_audit.md)\n\n")
        fails = aud[~aud["pass"]]
        if len(fails):
            f.write(_md(fails) + "\n\n")
        if st:
            f.write("## Validation self-tests\n\n" + "\n".join(
                f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in st.items()) + "\n")

    print(t9[["Response", "report"]].to_string(index=False))
    print(reg_df.to_string(index=False))
    print(f"Audit: {int(aud['pass'].sum())}/{len(aud)} passed")
    if st:
        print("Self-tests:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in st.items()))
    print(f"Results written to {out.resolve()}")
    return 0 if (not st or all(st.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
