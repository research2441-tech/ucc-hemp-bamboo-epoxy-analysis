"""
Statistical core of the pipeline (Section 3.8).

Implements exactly the procedures stated in the revised Methods:
  * descriptive statistics: mean, SD, CV and t-based 95 % confidence interval
  * one-way ANOVA with µCC loading as the single factor, reported as
    F(df_between, df_within), p and partial eta-squared (plus omega-squared)
  * assumption checks: Bartlett (from summary statistics), Brown-Forsythe /
    Levene and Shapiro-Wilk (from specimen-level data), Welch ANOVA as a
    variance-robust confirmation
  * Tukey HSD post-hoc comparisons (Tukey-Kramer form, valid for unequal n)
    with a compact letter display, and Games-Howell as the unequal-variance
    alternative

All tests are available in two equivalent forms:
  - from summary statistics (mean, SD, n), which is exact for one-way ANOVA
    and Tukey HSD because both depend only on group means, group variances
    and group sizes;
  - from specimen-level observations, when specimen_data.csv is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def ci_from_summary(mean: float, sd: float, n: int, level: float = 0.95):
    """Two-sided t-based confidence interval of the mean."""
    half = stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    return mean - half, mean + half


def describe(x: Sequence[float], level: float = 0.95) -> dict:
    """Mean, SD (n-1), CV % and confidence interval of one group."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else float("nan")
    lo, hi = ci_from_summary(m, s, n, level) if n > 1 else (np.nan, np.nan)
    return {"n": n, "mean": m, "sd": s, "cv_pct": 100 * s / m if m else np.nan,
            "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------
# One-way ANOVA
# --------------------------------------------------------------------------
@dataclass
class AnovaResult:
    F: float
    df_between: int
    df_within: int
    p: float
    ss_between: float
    ss_within: float
    ms_within: float
    partial_eta_sq: float
    omega_sq: float
    cohens_f: float

    def apa(self) -> str:
        p = "< 0.001" if self.p < 0.001 else f"= {self.p:.3f}"
        return (f"F({self.df_between}, {self.df_within}) = {self.F:.2f}, p {p}, "
                f"partial η² = {self.partial_eta_sq:.2f}")

    def as_dict(self) -> dict:
        return asdict(self)


def _anova_core(means, sds, ns) -> AnovaResult:
    means = np.asarray(means, float)
    sds = np.asarray(sds, float)
    ns = np.asarray(ns, float)
    k = means.size
    N = ns.sum()
    grand = np.sum(ns * means) / N
    ssb = float(np.sum(ns * (means - grand) ** 2))
    ssw = float(np.sum((ns - 1) * sds ** 2))
    dfb, dfw = k - 1, int(N - k)
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = float(stats.f.sf(F, dfb, dfw))
    eta = ssb / (ssb + ssw)                       # = partial η² in a one-way design
    omega = (ssb - dfb * msw) / (ssb + ssw + msw)
    return AnovaResult(F=F, df_between=dfb, df_within=dfw, p=p, ss_between=ssb,
                       ss_within=ssw, ms_within=msw, partial_eta_sq=eta,
                       omega_sq=max(omega, 0.0), cohens_f=np.sqrt(eta / (1 - eta)))


def anova_from_summary(means, sds, ns) -> AnovaResult:
    """Exact one-way ANOVA from group means, SDs (n-1 denominator) and sizes."""
    return _anova_core(means, sds, ns)


def anova_from_raw(groups: Sequence[Sequence[float]]) -> AnovaResult:
    """One-way ANOVA from specimen-level observations."""
    g = [np.asarray(x, float)[~np.isnan(np.asarray(x, float))] for x in groups]
    return _anova_core([x.mean() for x in g], [x.std(ddof=1) for x in g], [x.size for x in g])


def welch_anova_from_summary(means, sds, ns) -> dict:
    """Welch's heteroscedastic one-way ANOVA."""
    m = np.asarray(means, float)
    v = np.asarray(sds, float) ** 2
    n = np.asarray(ns, float)
    k = m.size
    w = n / v
    mw = np.sum(w * m) / w.sum()
    a = np.sum(w * (m - mw) ** 2) / (k - 1)
    lam = np.sum((1 - w / w.sum()) ** 2 / (n - 1))
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = a / b
    df2 = (k ** 2 - 1) / (3 * lam)
    return {"F": float(F), "df_between": k - 1, "df_within": float(df2),
            "p": float(stats.f.sf(F, k - 1, df2))}


def bartlett_from_summary(sds, ns) -> dict:
    """Bartlett test of homogeneity of variance using only SDs and n."""
    v = np.asarray(sds, float) ** 2
    n = np.asarray(ns, float)
    k = v.size
    N = n.sum()
    sp2 = np.sum((n - 1) * v) / (N - k)
    num = (N - k) * np.log(sp2) - np.sum((n - 1) * np.log(v))
    den = 1 + (np.sum(1 / (n - 1)) - 1 / (N - k)) / (3 * (k - 1))
    T = num / den
    return {"statistic": float(T), "df": k - 1, "p": float(stats.chi2.sf(T, k - 1))}


def assumption_checks_raw(groups) -> dict:
    """Shapiro-Wilk per group and Brown-Forsythe (median-centred Levene)."""
    g = [np.asarray(x, float)[~np.isnan(np.asarray(x, float))] for x in groups]
    sw = [float(stats.shapiro(x).pvalue) if x.size >= 3 else np.nan for x in g]
    bf = stats.levene(*g, center="median")
    return {"shapiro_p_min": float(np.nanmin(sw)), "shapiro_p": sw,
            "brown_forsythe_W": float(bf.statistic), "brown_forsythe_p": float(bf.pvalue)}


# --------------------------------------------------------------------------
# Post-hoc comparisons
# --------------------------------------------------------------------------
def tukey_from_summary(labels, means, sds, ns, alpha: float = 0.05) -> list[dict]:
    """Tukey HSD (Tukey-Kramer) pairwise comparisons from summary statistics."""
    an = _anova_core(means, sds, ns)
    k = len(means)
    qcrit = stats.studentized_range.ppf(1 - alpha, k, an.df_within)
    out = []
    for i, j in combinations(range(k), 2):
        se = np.sqrt(an.ms_within / 2 * (1 / ns[i] + 1 / ns[j]))
        diff = means[j] - means[i]
        q = abs(diff) / se
        p = float(stats.studentized_range.sf(q, k, an.df_within))
        out.append({"group_a": labels[i], "group_b": labels[j], "mean_diff": diff,
                    "hsd": qcrit * se, "ci_low": diff - qcrit * se,
                    "ci_high": diff + qcrit * se, "q": q, "p_adj": p,
                    "significant": p < alpha})
    return out


def games_howell_from_summary(labels, means, sds, ns, alpha: float = 0.05) -> list[dict]:
    """Games-Howell pairwise comparisons (no equal-variance assumption)."""
    k = len(means)
    v = np.asarray(sds, float) ** 2
    n = np.asarray(ns, float)
    out = []
    for i, j in combinations(range(k), 2):
        a, b = v[i] / n[i], v[j] / n[j]
        se = np.sqrt((a + b) / 2)
        df = (a + b) ** 2 / (a ** 2 / (n[i] - 1) + b ** 2 / (n[j] - 1))
        diff = means[j] - means[i]
        q = abs(diff) / se
        p = float(stats.studentized_range.sf(q, k, df))
        out.append({"group_a": labels[i], "group_b": labels[j], "mean_diff": diff,
                    "q": q, "df": df, "p_adj": p, "significant": p < alpha})
    return out


def tukey_from_raw(labels, groups, alpha: float = 0.05) -> list[dict]:
    """Tukey HSD from specimen-level data (delegates to the summary form)."""
    g = [np.asarray(x, float)[~np.isnan(np.asarray(x, float))] for x in groups]
    return tukey_from_summary(labels, [x.mean() for x in g], [x.std(ddof=1) for x in g],
                              [x.size for x in g], alpha)


def compact_letter_display(labels, means, pairwise: list[dict]) -> dict:
    """
    Compact letter display by the insert-absorb algorithm (Piepho, 2004).
    Groups sharing a letter are not significantly different. Letter 'a' is
    given to the group set containing the highest mean.
    """
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cols = [set(range(k))]
    for r in pairwise:
        if not r["significant"]:
            continue
        i, j = idx[r["group_a"]], idx[r["group_b"]]
        new = []
        for c in cols:
            if i in c and j in c:
                new.append(c - {i})
                new.append(c - {j})
            else:
                new.append(c)
        # absorb: drop columns contained in another column
        uniq = []
        for c in new:
            if c not in uniq:
                uniq.append(c)
        cols = [c for c in uniq if not any(c < d for d in uniq)]
    order = sorted(cols, key=lambda c: -max(means[i] for i in c))
    letters = {lab: "" for lab in labels}
    for li, c in enumerate(order):
        for i in sorted(c):
            letters[labels[i]] += "abcdefghijklmnopqrstuvwxyz"[li]
    return letters


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference
