"""
Numerical consistency audit of the manuscript.

Every value quoted in the Abstract and in Sections 4.1-4.12 is checked
against the table it is taken from, duplicated tables are cross-checked,
reported CV and 95 % CI columns are recomputed from mean, SD and n, and
every ordinal statement in the Discussion (maximum, minimum, increase,
decrease) is tested against the tabulated means.

Run on its own:  python consistency_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from stats_core import ci_from_summary

TOL = 1e-9


def load(path=config.REPORTED_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"load_level": str})
    df["load_level"] = df["load_level"].fillna("")
    df["ucc_wt"] = df["ucc_wt"].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    return df


def value(df, response, wt, level="", table=None, field="mean"):
    q = (df.response == response) & (df.ucc_wt.astype(str) == str(wt)) & (df.load_level == level)
    if table:
        q &= df.table == table
    else:
        q &= df.table != "T10"
    sel = df.loc[q, field]
    return float(sel.iloc[0]) if len(sel) else np.nan


# Values of Table 1 (summary table) that must equal Tables 4-6 and Table 10.
TABLE1 = {
    "tensile_strength": [(82, 4), (88, 4), (94, 3), (98, 4), (89, 5)],
    "flexural_strength": [(112, 5), (119, 5), (127, 4), (132, 5), (121, 6)],
    "impact_strength": [(38, 3), (42, 3), (47, 2), (50, 3), (43, 3)],
}

# CV (%) and 95 % CI columns printed in Tables 4-6.
REPORTED_CV_CI = {
    "tensile_strength": [(4.9, 79.1, 84.9), (4.5, 85.1, 90.9), (3.2, 91.9, 96.1),
                         (4.1, 95.1, 100.9), (5.6, 85.4, 92.6)],
    "flexural_strength": [(4.5, 108.4, 115.6), (4.2, 115.4, 122.6), (3.1, 124.1, 129.9),
                          (3.8, 128.4, 135.6), (5.0, 116.7, 125.3)],
    "impact_strength": [(7.9, None, None), (7.1, None, None), (4.3, None, None),
                        (6.0, None, None), (7.0, None, None)],
}

# Values quoted in the text of the revised manuscript: (location, response, wt, level, mean, sd)
TEXT_VALUES = [
    ("Abstract", "tensile_strength", 0, "", 82, 4), ("Abstract", "tensile_strength", 3, "", 98, 4),
    ("Abstract", "flexural_strength", 0, "", 112, 5), ("Abstract", "flexural_strength", 3, "", 132, 5),
    ("Abstract", "impact_strength", 0, "", 38, 3), ("Abstract", "impact_strength", 3, "", 50, 3),
    ("Abstract", "agglomerate_size", 5, "", 18.6, 2.7), ("Abstract", "void_fraction", 5, "", 3.40, 0.38),
    ("4.1", "tensile_strength", 5, "", 89, 5), ("4.1", "flexural_strength", 5, "", 121, 6),
    ("4.1", "impact_strength", 5, "", 43, 3),
    ("4.2", "agglomerate_size", 0, "", 6.2, 0.8), ("4.2", "agglomerate_size", 3, "", 10.8, 1.4),
    ("4.2", "agglomerate_area_fraction", 0, "", 1.8, 0.3),
    ("4.2", "agglomerate_area_fraction", 3, "", 4.7, 0.6),
    ("4.2", "agglomerate_area_fraction", 5, "", 8.9, 1.1),
    ("4.3", "void_fraction", 0, "", 2.10, 0.25), ("4.3", "void_fraction", 2, "", 1.80, 0.20),
    ("4.3", "void_fraction", 3, "", 2.00, 0.24), ("4.3", "void_fraction", 5, "", 3.40, 0.38),
    ("4.3", "fiber_spacing_variation", 0, "", 18.4, 2.1),
    ("4.3", "fiber_spacing_variation", 3, "", 14.8, 1.6),
    ("4.3", "fiber_spacing_variation", 5, "", 21.7, 2.6),
    ("4.4", "interfacial_gap_index", 0, "", 0.42, 0.06), ("4.4", "interfacial_gap_index", 2, "", 0.28, 0.04),
    ("4.4", "interfacial_gap_index", 3, "", 0.31, 0.05), ("4.4", "interfacial_gap_index", 5, "", 0.58, 0.08),
    ("4.4", "debonded_interface_frequency", 0, "", 21.0, 2.6),
    ("4.4", "debonded_interface_frequency", 2, "", 13.0, 1.8),
    ("4.4", "debonded_interface_frequency", 5, "", 28.0, 3.2),
    ("4.4", "pullout_frequency", 0, "", 18.0, 2.4), ("4.4", "pullout_frequency", 2, "", 11.0, 1.7),
    ("4.4", "pullout_frequency", 5, "", 24.0, 3.1),
    ("4.4", "mean_pullout_length", 0, "", 146, 18), ("4.4", "mean_pullout_length", 2, "", 104, 13),
    ("4.5", "tensile_modulus", 0, "", 5.8, 0.3), ("4.5", "tensile_modulus", 3, "", 6.8, 0.4),
    ("4.5", "flexural_modulus", 0, "", 6.4, 0.3), ("4.5", "flexural_modulus", 3, "", 7.5, 0.4),
    ("4.6", "crack_density", 3, "25", 0.35, 0.05), ("4.6", "crack_density", 0, "25", 0.62, 0.09),
    ("4.6", "crack_density", 5, "25", 0.91, 0.12), ("4.6", "crack_density", 3, "90", 1.73, 0.20),
    ("4.6", "crack_density", 0, "90", 2.72, 0.29), ("4.6", "crack_density", 5, "90", 3.08, 0.34),
    ("4.10", "water_absorption_720h", 0, "", 8.60, 0.50),
    ("4.10", "water_absorption_720h", 2, "", 7.70, 0.40),
    ("4.10", "water_absorption_720h", 3, "", 7.90, 0.50),
    ("4.10", "water_absorption_720h", 5, "", 9.80, 0.60),
    ("4.11", "degradation_onset", 0, "", 312, 4), ("4.11", "degradation_onset", 3, "", 329, 4),
    ("4.11", "max_degradation_temperature", 0, "", 356, 5),
    ("4.11", "max_degradation_temperature", 3, "", 372, 5),
    ("4.11", "glass_transition_temperature", 0, "", 78, 2),
    ("4.11", "glass_transition_temperature", 3, "", 83, 2),
    ("4.11", "glass_transition_temperature", 5, "", 80, 3),
    ("4.11", "crack_density_post", 5, "", 4.52, 0.43), ("4.11", "crack_density_post", 2, "", 2.61, 0.27),
    ("4.11", "interfacial_gap_index_post", 5, "", 0.76, 0.09),
    ("4.11", "interfacial_gap_index_post", 2, "", 0.39, 0.05),
]


def _means(df, response, level="", table=None):
    q = (df.response == response) & (df.load_level == level)
    q &= (df.table == table) if table else (df.table != "T10")
    sub = df[q]
    return dict(zip(sub.ucc_wt.astype(str), sub["mean"].astype(float)))


def _argmax(d):
    return max(d, key=d.get)


def _argmin(d):
    return min(d, key=d.get)


def ordinal_claims(df):
    """(location, statement, bool) for every ordinal statement in the Discussion."""
    c = []
    for r in config.MECHANICAL_RESPONSES + ["tensile_modulus", "flexural_modulus"]:
        c.append(("4.1/4.5", f"{r} is highest at 3 wt%", _argmax(_means(df, r)) == "3"))
    tg = _means(df, "glass_transition_temperature")
    c.append(("4.11", "Tg is highest at 3 wt%", _argmax(tg) == "3"))
    c.append(("4.11", "Tg at 5 wt% is lower than at 3 wt%", tg["5"] < tg["3"]))
    c.append(("4.11", "Tg at 5 wt% is higher than the unfilled laminate", tg["5"] > tg["0"]))
    for r in ["degradation_onset", "max_degradation_temperature"]:
        c.append(("4.11", f"{r} is highest at 3 wt%", _argmax(_means(df, r)) == "3"))
    vf = _means(df, "void_fraction")
    c.append(("4.3", "void fraction is highest at 5 wt%", _argmax(vf) == "5"))
    c.append(("4.3", "void fraction is lowest at 2 wt%", _argmin(vf) == "2"))
    c.append(("4.3", "void fraction at 5 wt% exceeds every other loading",
              all(vf["5"] > vf[k] for k in vf if k != "5")))
    for r in ["agglomerate_size", "agglomerate_area_fraction", "agglomerate_number_density"]:
        v = [_means(df, r)[str(w)] for w in config.UCC_LOADINGS_WT]
        c.append(("4.2", f"{r} increases monotonically with loading", bool(np.all(np.diff(v) > 0))))
    fsv = [_means(df, "fiber_spacing_variation")[str(w)] for w in config.UCC_LOADINGS_WT]
    c.append(("4.3", "fibre-spacing variation decreases to 3 wt% then rises at 5 wt%",
              bool(np.all(np.diff(fsv[:4]) < 0) and fsv[4] > fsv[3])))
    for r in ["interfacial_gap_index", "debonded_interface_frequency", "pullout_frequency"]:
        c.append(("4.4", f"{r} is lowest at 2 wt% and highest at 5 wt%",
                  _argmin(_means(df, r)) == "2" and _argmax(_means(df, r)) == "5"))
    for lev in ["25", "50", "75", "90"]:
        cd = _means(df, "crack_density", lev)
        c.append(("4.6", f"crack density at {lev}% load: lowest at 3 wt%, highest at 5 wt%",
                  _argmin(cd) == "3" and _argmax(cd) == "5"))
    wa = _means(df, "water_absorption_720h")
    c.append(("4.10", "water absorption lowest at 2 wt%, highest at 5 wt%",
              _argmin(wa) == "2" and _argmax(wa) == "5"))
    rt = _means(df, "tensile_retention_720h")
    c.append(("4.10/4.12", "tensile retention highest at 2 wt%, lowest at 5 wt%",
              _argmax(rt) == "2" and _argmin(rt) == "5"))
    for r in ["void_fraction_post", "interfacial_gap_index_post", "crack_density_post",
              "pullout_frequency_post", "delaminated_area_post"]:
        c.append(("4.11", f"{r} is highest at 5 wt%", _argmax(_means(df, r)) == "5"))
    sysmeans = df[(df.table == "T10") & (df.response == "tensile_strength")].set_index("system")["mean"]
    c.append(("4.9", "tensile: neat < hemp < bamboo < hemp-bamboo",
              sysmeans["neat epoxy"] < sysmeans["hemp/epoxy"] < sysmeans["bamboo/epoxy"]
              < sysmeans["hemp-bamboo/epoxy"]))
    for r in config.MECHANICAL_RESPONSES:
        sub = df[(df.response == r) & (df.table != "T10")].set_index("ucc_wt")
        cv = 100 * sub["sd"] / sub["mean"]
        c.append(("4.8", f"{r}: CV at 2-3 wt% below CV at 0 and 5 wt%",
                  max(cv["2"], cv["3"]) < min(cv["0"], cv["5"])))
    ts = _means(df, "tensile_strength")
    c.append(("4.5", "tensile strength gain at 3 wt% is 19.5 %",
              round(100 * (ts["3"] - ts["0"]) / ts["0"], 1) == 19.5))
    return c


def run(df=None, outdir: Path | None = None) -> pd.DataFrame:
    df = load() if df is None else df
    rec = []

    # 1. missing table cells
    for _, r in df[df["mean"].isna()].iterrows():
        rec.append(("missing cell", f"{r.table}", f"{r.response} at {r.ucc_wt} wt%",
                    "value present", "missing", False))

    # 2. Table 1 vs Tables 4-6 and Table 10
    for resp, vals in TABLE1.items():
        for w, (m, s) in zip(config.UCC_LOADINGS_WT, vals):
            for tab in [None, "T10"]:
                mm = value(df, resp, w, table=tab)
                ss = value(df, resp, w, table=tab, field="sd")
                ok = abs(mm - m) < TOL and abs(ss - s) < TOL
                rec.append(("cross-table", f"T1 vs {'T10' if tab else 'T4-T6'}",
                            f"{resp} {w} wt%", f"{m} ± {s}", f"{mm:g} ± {ss:g}", ok))

    # 3. CV and CI columns
    for resp, rows in REPORTED_CV_CI.items():
        for w, (cv, lo, hi) in zip(config.UCC_LOADINGS_WT, rows):
            m, s = value(df, resp, w), value(df, resp, w, field="sd")
            n = int(value(df, resp, w, field="n"))
            cvc = 100 * s / m
            rec.append(("CV column", "T4-T6", f"{resp} {w} wt%", f"{cv}", f"{cvc:.1f}",
                        abs(round(cvc, 1) - cv) <= 0.1))
            if lo is not None:
                l2, h2 = ci_from_summary(m, s, n, config.CI_LEVEL)
                rec.append(("95% CI column", "T4-T5", f"{resp} {w} wt% (n={n})",
                            f"{lo}-{hi}", f"{l2:.1f}-{h2:.1f}",
                            abs(l2 - lo) <= 0.15 and abs(h2 - hi) <= 0.15))

    # 4. text values vs tables
    for loc, resp, w, lev, m, s in TEXT_VALUES:
        mm = value(df, resp, w, lev)
        ss = value(df, resp, w, lev, field="sd")
        ok = abs(mm - m) < 1e-6 and abs(ss - s) < 1e-6
        rec.append(("text vs table", loc, f"{resp} {w} wt% {lev}".strip(),
                    f"{m} ± {s}", f"{mm:g} ± {ss:g}", ok))

    # 5. ordinal statements
    for loc, stmt, ok in ordinal_claims(df):
        rec.append(("ordinal statement", loc, stmt, "true", str(bool(ok)).lower(), bool(ok)))

    out = pd.DataFrame(rec, columns=["check", "location", "item", "expected", "found", "pass"])
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / "consistency_audit.csv", index=False)
        with open(outdir / "consistency_audit.md", "w", encoding="utf-8") as f:
            f.write("# Numerical consistency audit\n\n")
            f.write(f"Checks run: {len(out)}  |  passed: {int(out['pass'].sum())}  |  "
                    f"failed: {int((~out['pass']).sum())}\n\n")
            for chk, g in out.groupby("check", sort=False):
                f.write(f"## {chk} ({int(g['pass'].sum())}/{len(g)} passed)\n\n")
                f.write("| Location | Item | Expected | Found | Pass |\n|---|---|---|---|---|\n")
                for _, r in g.iterrows():
                    f.write(f"| {r.location} | {r.item} | {r.expected} | {r.found} | "
                            f"{'yes' if r['pass'] else '**NO**'} |\n")
                f.write("\n")
    return out


if __name__ == "__main__":
    res = run(outdir=config.DEFAULT_OUTDIR)
    print(f"{int(res['pass'].sum())}/{len(res)} checks passed")
    bad = res[~res["pass"]]
    if len(bad):
        print(bad.to_string(index=False))
    sys.exit(0 if bad.empty else 1)
