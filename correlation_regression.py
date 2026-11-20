"""
Microstructure-interface-property relationships (Section 3.7, Section 4.7).

  * Pearson correlation (normally distributed, linear) and Spearman rank
    correlation (non-normal or monotonic non-linear), each with a Fisher-z
    95 % confidence interval
  * multiple linear regression of tensile strength on standardised
    microstructural and interfacial predictors, with standardised
    coefficients (beta), p-values, adjusted R^2 and variance-inflation factors
  * relative importance of each predictor as its share of R^2 by the
    Lindeman-Merenda-Gold (LMG) decomposition, which averages the R^2 gain of
    a predictor over all orders of entry and therefore sums exactly to R^2

Specimen-level analysis requires specimen_data.csv (one row per specimen,
microstructure measured on sections cut from the tested coupon). Without it,
correlations are evaluated on the five formulation means and are labelled
as indicative, because n = 5.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["agglomerate_area_fraction", "interfacial_gap_index", "void_fraction",
              "fiber_distribution_uniformity", "pullout_frequency"]
TARGETS = ["tensile_strength", "flexural_strength", "impact_strength"]


def correlation(x, y, method="pearson", level=0.95) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = x.size
    r, p = (stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y))
    r = float(r)
    if n > 3 and abs(r) < 1:
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        zc = stats.norm.ppf(0.5 + level / 2)
        lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
    else:
        lo = hi = np.nan
    return {"method": method, "n": n, "r": r, "p": float(p), "ci_low": float(lo), "ci_high": float(hi)}


def correlation_table(df: pd.DataFrame, predictors=PREDICTORS, targets=TARGETS) -> pd.DataFrame:
    rows = []
    for t in targets:
        for pcol in predictors:
            if t not in df or pcol not in df:
                continue
            x, y = df[pcol], df[t]
            normal = (stats.shapiro(x.dropna()).pvalue > 0.05 and stats.shapiro(y.dropna()).pvalue > 0.05
                      if min(x.notna().sum(), y.notna().sum()) >= 3 else True)
            for m in ("pearson", "spearman"):
                r = correlation(x, y, m)
                r.update(predictor=pcol, target=t, recommended=(m == "pearson") == normal)
                rows.append(r)
    return pd.DataFrame(rows)


def _r2(X, y):
    if X.shape[1] == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    return 1 - res @ res / np.sum((y - y.mean()) ** 2)


def lmg_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LMG (Shapley) decomposition of R^2 across predictors."""
    p = X.shape[1]
    cache = {}

    def r2(S):
        key = tuple(sorted(S))
        if key not in cache:
            cache[key] = _r2(X[:, list(key)], y)
        return cache[key]

    imp = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            w = factorial(size) * factorial(p - size - 1) / factorial(p)
            for S in combinations(others, size):
                imp[j] += w * (r2(S + (j,)) - r2(S))
    return imp


def standardized_regression(df: pd.DataFrame, target="tensile_strength", predictors=PREDICTORS) -> tuple[pd.DataFrame, dict]:
    d = df[[target] + predictors].dropna()
    n, p = len(d), len(predictors)
    if n <= p + 2:
        raise ValueError(f"Regression needs more than {p + 2} specimens; found {n}.")
    Z = (d - d.mean()) / d.std(ddof=1)
    X, y = Z[predictors].to_numpy(), Z[target].to_numpy()
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = n - p - 1
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    adj = 1 - (1 - r2) * (n - 1) / dof
    vif = [1 / (1 - _r2(np.delete(X, j, axis=1), X[:, j])) for j in range(p)]
    lmg = lmg_importance(X, y)
    table = pd.DataFrame({"predictor": predictors, "beta": beta[1:], "se": se[1:], "t": t[1:],
                          "p": pv[1:], "vif": vif, "lmg_r2_share": lmg,
                          "relative_contribution_pct": 100 * lmg / r2})
    F = (r2 / p) / ((1 - r2) / dof)
    model = {"n": n, "r2": r2, "adj_r2": adj, "F": F, "df1": p, "df2": dof,
             "p_model": float(stats.f.sf(F, p, dof))}
    return table, model


def add_uniformity(df: pd.DataFrame) -> pd.DataFrame:
    """Fibre-distribution uniformity = 100 - fibre-spacing variation (%)."""
    df = df.copy()
    if "fiber_spacing_variation" in df and "fiber_distribution_uniformity" not in df:
        df["fiber_distribution_uniformity"] = 100 - df["fiber_spacing_variation"]
    return df


def selftest(verbose=True) -> bool:
    rng = np.random.default_rng(7)
    n = 150
    X = rng.normal(size=(n, 3))
    y = 0.6 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(y=y)
    tab, mod = standardized_regression(df, "y", ["a", "b", "c"])
    ok = abs(tab.lmg_r2_share.sum() - mod["r2"]) < 1e-10
    ok &= tab.set_index("predictor").lmg_r2_share.idxmax() == "a"
    ok &= tab.set_index("predictor").loc["c", "p"] > 0.01
    if verbose:
        print(tab.round(3).to_string(index=False))
        print(f"R2 {mod['r2']:.3f} = sum LMG {tab.lmg_r2_share.sum():.3f} -> {'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
