"""
TGA and DSC curve processing (Section 3.11, Table 11).

TGA  (mass % vs temperature)
  * T5            temperature at 5 % mass loss
  * T_onset       extrapolated onset: intersection of the pre-degradation
                  baseline tangent with the tangent at the DTG inflection point
                  (ISO 11358-1)
  * T_max         peak of the derivative thermogravimetric (DTG) curve
  * residue       mass % remaining at 600 °C

DSC  (heat flow vs temperature, second heating scan)
  * Tg midpoint   temperature at half the step height between the extrapolated
                  pre- and post-transition baselines (ISO 11357-2 / ASTM E1356)
  * Tg onset and Tg inflection are reported alongside

Instrument exports are read as two-column CSV files. Batch naming:
  TGA_L<wt>_R<replicate>.csv   columns: temperature_C, mass_pct
  DSC_L<wt>_R<replicate>.csv   columns: temperature_C, heat_flow_W_per_g

The test conditions in config.TGA_CONDITIONS and config.DSC_CONDITIONS are
written into every output so the processing record carries the heating
rate, atmosphere, purge flow and sample-mass range stated in Section 3.11.

Usage
  python thermal_analysis.py --selftest
  python thermal_analysis.py --curves path/to/exports --out results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config

NAME_RE = re.compile(r"(?P<kind>TGA|DSC)_L(?P<wt>\d+(?:\.\d+)?)_R(?P<rep>\d+)", re.I)


def _uniform(T, y, step=0.5):
    order = np.argsort(T)
    T, y = np.asarray(T, float)[order], np.asarray(y, float)[order]
    Tg = np.arange(T[0], T[-1], step)
    return Tg, np.interp(Tg, T, y)


def _smooth_derivative(T, y, window_C=10.0):
    step = T[1] - T[0]
    win = max(5, int(window_C / step) | 1)
    return savgol_filter(y, win, 3, deriv=1, delta=step), savgol_filter(y, win, 3)


# --------------------------------------------------------------------------
# TGA
# --------------------------------------------------------------------------
def analyse_tga(T, mass, baseline_window=(150.0, 250.0)) -> dict:
    T, m = _uniform(T, mass)
    m = 100.0 * m / m[np.searchsorted(T, min(T[0] + 5, 120.0))]   # normalise after moisture
    dm, ms = _smooth_derivative(T, m)
    search = T > baseline_window[0]
    i_pk = np.argmin(np.where(search, dm, np.inf))                 # steepest mass loss
    t_max = T[i_pk]
    lo, hi = baseline_window
    hi = min(hi, t_max - 30)
    sel = (T >= lo) & (T <= hi)
    a_b, b_b = np.polyfit(T[sel], ms[sel], 1)
    slope, inter = dm[i_pk], ms[i_pk] - dm[i_pk] * t_max
    t_onset = (inter - b_b) / (a_b - slope)
    t5 = T[np.argmax(ms <= 95.0)]
    residue = float(np.interp(600.0, T, ms)) if T[-1] >= 600 else np.nan
    return {"T5_C": float(t5), "T_onset_C": float(t_onset), "T_max_C": float(t_max),
            "max_mass_loss_rate_pct_per_C": float(-dm[i_pk]), "residue_600C_pct": residue}


# --------------------------------------------------------------------------
# DSC
# --------------------------------------------------------------------------
def analyse_dsc(T, heat_flow, search=(40.0, 150.0), base_offset=(8.0, 25.0)) -> dict:
    T, h = _uniform(T, heat_flow, step=0.2)
    dh, hs = _smooth_derivative(T, h, window_C=4.0)
    win = (T >= search[0]) & (T <= search[1])
    # the glass transition is the largest-magnitude step in the window
    i_inf = np.flatnonzero(win)[np.argmax(np.abs(dh[win]))]
    t_inf = T[i_inf]
    d1, d2 = base_offset
    pre = (T >= t_inf - d2) & (T <= t_inf - d1)
    post = (T >= t_inf + d1) & (T <= t_inf + d2)
    a1, b1 = np.polyfit(T[pre], hs[pre], 1)
    a2, b2 = np.polyfit(T[post], hs[post], 1)
    mid_line = 0.5 * ((a1 * T + b1) + (a2 * T + b2))
    region = (T > t_inf - d2) & (T < t_inf + d2)
    diff = (hs - mid_line)[region]
    Tr = T[region]
    cross = np.flatnonzero(np.diff(np.sign(diff)) != 0)
    j = cross[np.argmin(np.abs(Tr[cross] - t_inf))]
    tg_mid = Tr[j] - diff[j] * (Tr[j + 1] - Tr[j]) / (diff[j + 1] - diff[j])
    s, c = dh[i_inf], hs[i_inf] - dh[i_inf] * t_inf
    tg_on = (c - b1) / (a1 - s)
    dcp = abs((a2 * tg_mid + b2) - (a1 * tg_mid + b1))
    return {"Tg_midpoint_C": float(tg_mid), "Tg_onset_C": float(tg_on),
            "Tg_inflection_C": float(t_inf), "step_height": float(dcp)}


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------
def analyse_folder(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in sorted(Path(folder).glob("*.csv")):
        m = NAME_RE.search(f.stem)
        if not m:
            continue
        d = pd.read_csv(f)
        T, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
        res = analyse_tga(T, y) if m["kind"].upper() == "TGA" else analyse_dsc(T, y)
        res.update(file=f.name, technique=m["kind"].upper(), ucc_wt=float(m["wt"]), replicate=int(m["rep"]))
        rows.append(res)
    reps = pd.DataFrame(rows)
    if reps.empty:
        raise FileNotFoundError(f"No TGA_/DSC_ curve files found in {folder}")
    num = [c for c in reps.columns if c.endswith(("_C", "_pct", "_per_C", "height"))]
    summ = reps.groupby(["technique", "ucc_wt"])[num].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    return reps, summ.reset_index()


# --------------------------------------------------------------------------
# Validation with analytic curves of known characteristic temperatures
# --------------------------------------------------------------------------
def synthetic_tga(t_max=350.0, width=12.0, residue=15.0):
    T = np.arange(30, 700, 1.0)
    moisture = 1.5 / (1 + np.exp(-(T - 80) / 8))
    m = 100 - moisture - (100 - residue - 1.5) / (1 + np.exp(-(T - t_max) / width))
    return T, m


def synthetic_dsc(tg=80.0, width=2.5, step=0.12):
    T = np.arange(25, 200, 0.5)
    h = -0.20 - 0.0008 * (T - 25) - step / (1 + np.exp(-(T - tg) / width))
    return T, h


def selftest(verbose=True) -> bool:
    ok = True
    for tm in (340.0, 356.0, 372.0):
        r = analyse_tga(*synthetic_tga(t_max=tm))
        good = abs(r["T_max_C"] - tm) <= 1.0 and r["T_onset_C"] < tm and r["T5_C"] < r["T_onset_C"] + 15
        ok &= good
        if verbose:
            print(f"TGA true Tmax {tm:.0f} -> Tmax {r['T_max_C']:.1f}, onset {r['T_onset_C']:.1f}, "
                  f"T5 {r['T5_C']:.1f} °C  {'PASS' if good else 'FAIL'}")
    for tg in (78.0, 80.0, 83.0):
        r = analyse_dsc(*synthetic_dsc(tg=tg))
        good = abs(r["Tg_midpoint_C"] - tg) <= 0.5
        ok &= good
        if verbose:
            print(f"DSC true Tg {tg:.0f} -> midpoint {r['Tg_midpoint_C']:.2f}, onset "
                  f"{r['Tg_onset_C']:.2f} °C  {'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", help="folder of TGA_/DSC_ CSV exports")
    ap.add_argument("--out", default=str(config.DEFAULT_OUTDIR))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.curves:
        raise SystemExit(0 if selftest() else 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reps, summ = analyse_folder(a.curves)
    reps.to_csv(out / "thermal_replicates.csv", index=False)
    summ.to_csv(out / "thermal_by_formulation.csv", index=False)
    with open(out / "thermal_conditions.json", "w") as f:
        json.dump({"TGA": config.TGA_CONDITIONS, "DSC": config.DSC_CONDITIONS}, f, indent=2)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
