"""
Verification tests for every computational component of the pipeline.

    python -m pytest -q test_pipeline.py      (or)      python test_pipeline.py

Each statistical routine is checked against an independent SciPy
implementation on specimen-level data constructed to have exactly the
tabulated mean and SD; the image, thermal and diffusion routines are checked
against inputs of known ground truth.
"""

import numpy as np
import pandas as pd
from scipy import stats

import config
import consistency_audit as audit
import correlation_regression as cr
import image_analysis as ia
import moisture_retention as mr
import stats_core as sc
import thermal_analysis as ta

LAB = ["0", "1", "2", "3", "5"]
TS_M = [82, 88, 94, 98, 89]
TS_S = [4, 4, 3, 4, 5]
N = [10] * 5


def _exact_groups(means, sds, n, seed=0):
    """Observations whose sample mean and SD equal the given values exactly."""
    rng = np.random.default_rng(seed)
    out = []
    for m, s in zip(means, sds):
        z = rng.normal(size=n)
        z = (z - z.mean()) / z.std(ddof=1)
        out.append(m + s * z)
    return out


def test_anova_summary_equals_scipy_raw():
    g = _exact_groups(TS_M, TS_S, 10)
    a = sc.anova_from_summary(TS_M, TS_S, N)
    ref = stats.f_oneway(*g)
    assert np.isclose(a.F, ref.statistic) and np.isclose(a.p, ref.pvalue)
    assert (a.df_between, a.df_within) == (4, 45)


def test_tukey_summary_equals_scipy_tukey_hsd():
    g = _exact_groups(TS_M, TS_S, 10)
    ref = stats.tukey_hsd(*g).pvalue
    for r in sc.tukey_from_summary(LAB, TS_M, TS_S, N):
        i, j = LAB.index(r["group_a"]), LAB.index(r["group_b"])
        assert np.isclose(r["p_adj"], ref[i, j], atol=1e-4)


def test_welch_and_bartlett_match_scipy():
    g = _exact_groups(TS_M, TS_S, 10)
    assert np.isclose(sc.bartlett_from_summary(TS_S, N)["p"], stats.bartlett(*g).pvalue)
    w = sc.welch_anova_from_summary(TS_M, TS_S, N)
    assert w["p"] < 0.001 and w["df_between"] == 4


def test_compact_letter_display():
    letters = sc.compact_letter_display(LAB, TS_M, sc.tukey_from_summary(LAB, TS_M, TS_S, N))
    assert letters == {"0": "d", "1": "c", "2": "ab", "3": "a", "5": "bc"}
    # 3 wt% differs from the unfilled laminate; 1 and 5 wt% share a group
    assert not set(letters["3"]) & set(letters["0"])
    assert set(letters["1"]) & set(letters["5"])


def test_confidence_interval_matches_table4():
    lo, hi = sc.ci_from_summary(82, 4, 10)
    assert round(lo, 1) == 79.1 and round(hi, 1) == 84.9


def test_consistency_audit_only_flags_missing_cells():
    res = audit.run()
    fails = res[~res["pass"]]
    assert set(fails["check"]) <= {"missing cell"}


def test_image_analysis_recovers_phantom_truth():
    img, fibre, truth = ia.make_phantom(seed=11)
    r = ia.quantify_field(img, fibre)
    assert abs(r["void_fraction_pct"] - truth["void_fraction_pct"]) < 0.15
    assert abs(r["agglomerate_area_fraction_pct"] - truth["agglomerate_area_fraction_pct"]) < 0.25


def test_void_fraction_equation():
    m = np.zeros((100, 100), bool)
    m[:10, :34] = True
    assert np.isclose(ia.void_fraction(m), 3.40)


def test_thermal_characteristic_temperatures():
    r = ta.analyse_tga(*ta.synthetic_tga(t_max=372.0))
    assert abs(r["T_max_C"] - 372.0) <= 1.0
    d = ta.analyse_dsc(*ta.synthetic_dsc(tg=83.0))
    assert abs(d["Tg_midpoint_C"] - 83.0) <= 0.5


def test_fickian_fit_and_equations():
    t = np.array(config.EXPOSURE_TIMES_H, float)
    M = mr.fick_model(t * 3600, 8.6, 2e-6, config.SPECIMEN_THICKNESS_MM)
    f = mr.fit_fick(t, M)
    assert abs(f["M_inf_pct"] - 8.6) < 0.05
    assert np.isclose(mr.water_absorption(10.0, 10.86), 8.6)
    assert np.isclose(mr.retention(100.0, 84.0), 84.0)


def test_lmg_shares_sum_to_r2():
    assert cr.selftest(verbose=False)


def test_reported_data_schema():
    df = pd.read_csv(config.REPORTED_DATA)
    assert {"table", "response", "ucc_wt", "mean", "sd", "n"} <= set(df.columns)
    assert (df["sd"] > 0).all() and (df["n"] >= 3).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}  {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
