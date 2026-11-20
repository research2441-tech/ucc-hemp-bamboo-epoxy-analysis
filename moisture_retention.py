"""
Water absorption, Fickian diffusion and property retention (Sections 3.10-3.11).

  W_t = (m_t - m_0) / m_0 x 100                                    (Eq. 2)
  R   = P_t / P_0 x 100                                            (Eq. 3)

The diffusion coefficient is obtained from the one-dimensional Fickian
solution for a plate of thickness h exposed on both faces (Crank, 1975):

  M_t / M_inf = 1 - (8 / pi^2) * sum_j exp(-D (2j+1)^2 pi^2 t / h^2) / (2j+1)^2

fitted by non-linear least squares, with the classical initial-slope
estimate D = pi * (h * k / (4 M_inf))^2 reported for comparison, where k is
the slope of M_t against sqrt(t) in the linear region (M_t / M_inf < 0.6).

Input for batch mode: CSV with columns  ucc_wt, specimen, time_h, mass_g
Usage
  python moisture_retention.py --selftest
  python moisture_retention.py --uptake water_uptake.csv --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import config


def water_absorption(m0, mt):
    """Eq. 2."""
    m0 = np.asarray(m0, float)
    return 100.0 * (np.asarray(mt, float) - m0) / m0


def retention(p0, pt):
    """Eq. 3."""
    return 100.0 * np.asarray(pt, float) / np.asarray(p0, float)


def retention_from_summary(mean0, sd0, mean_t, sd_t):
    """Retention and its SD by first-order error propagation of a ratio."""
    r = 100.0 * mean_t / mean0
    return r, r * np.sqrt((sd_t / mean_t) ** 2 + (sd0 / mean0) ** 2)


def fick_model(t_s, m_inf, D, h_mm, terms=60):
    j = np.arange(terms)[:, None]
    k = (2 * j + 1) ** 2
    s = np.sum(np.exp(-D * k * np.pi ** 2 * np.asarray(t_s)[None, :] / h_mm ** 2) / k, axis=0)
    return m_inf * (1 - 8 / np.pi ** 2 * s)


def fit_fick(time_h, uptake_pct, h_mm=config.SPECIMEN_THICKNESS_MM) -> dict:
    t = np.asarray(time_h, float) * 3600.0
    M = np.asarray(uptake_pct, float)
    m0 = max(M.max(), 1e-6)
    D0 = 1e-6
    (m_inf, D), cov = curve_fit(lambda tt, a, d: fick_model(tt, a, d, h_mm), t, M,
                                p0=[m0 * 1.05, D0], bounds=([0, 1e-12], [100, 1e-2]), maxfev=20000)
    perr = np.sqrt(np.diag(cov))
    pred = fick_model(t, m_inf, D, h_mm)
    ss_res, ss_tot = np.sum((M - pred) ** 2), np.sum((M - M.mean()) ** 2)
    lin = (M / m_inf < 0.6) & (t > 0)
    D_slope = np.nan
    if lin.sum() >= 2:
        k = np.polyfit(np.sqrt(t[lin]), M[lin], 1)[0]
        D_slope = np.pi * (h_mm * k / (4 * m_inf)) ** 2
    return {"M_inf_pct": float(m_inf), "M_inf_se": float(perr[0]), "D_mm2_per_s": float(D),
            "D_se": float(perr[1]), "D_initial_slope_mm2_per_s": float(D_slope),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def analyse_uptake(csv_path: str | Path, h_mm=config.SPECIMEN_THICKNESS_MM) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(csv_path)
    d = d.sort_values(["ucc_wt", "specimen", "time_h"])
    d["m0"] = d.groupby(["ucc_wt", "specimen"])["mass_g"].transform("first")
    d["uptake_pct"] = water_absorption(d["m0"], d["mass_g"])
    fits = []
    for (w, s), g in d.groupby(["ucc_wt", "specimen"]):
        r = fit_fick(g["time_h"], g["uptake_pct"], h_mm)
        r.update(ucc_wt=w, specimen=s, uptake_final_pct=float(g["uptake_pct"].iloc[-1]))
        fits.append(r)
    fits = pd.DataFrame(fits)
    summ = fits.groupby("ucc_wt")[["uptake_final_pct", "M_inf_pct", "D_mm2_per_s"]].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    return d, fits.merge(summ.reset_index(), on="ucc_wt", how="left")


def selftest(verbose=True) -> bool:
    ok = True
    t_h = np.array(config.EXPOSURE_TIMES_H, float)
    for m_inf, D in [(8.6, 2.0e-6), (7.7, 1.4e-6), (9.8, 2.8e-6)]:
        M = fick_model(t_h * 3600, m_inf, D, config.SPECIMEN_THICKNESS_MM)
        r = fit_fick(t_h, M)
        good = abs(r["M_inf_pct"] - m_inf) / m_inf < 0.01 and abs(r["D_mm2_per_s"] - D) / D < 0.02
        ok &= good
        if verbose:
            print(f"Fick true M_inf {m_inf}, D {D:.2e} -> {r['M_inf_pct']:.3f}, "
                  f"{r['D_mm2_per_s']:.3e} mm²/s  {'PASS' if good else 'FAIL'}")
    r = retention([82.0], [68.9])[0]
    ok &= abs(r - 84.02) < 0.01
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uptake", help="CSV with ucc_wt, specimen, time_h, mass_g")
    ap.add_argument("--thickness", type=float, default=config.SPECIMEN_THICKNESS_MM)
    ap.add_argument("--out", default=str(config.DEFAULT_OUTDIR))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.uptake:
        raise SystemExit(0 if selftest() else 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curves, fits = analyse_uptake(a.uptake, a.thickness)
    curves.to_csv(out / "water_uptake_curves.csv", index=False)
    fits.to_csv(out / "fickian_diffusion_fits.csv", index=False)
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
