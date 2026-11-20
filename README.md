# µCC hemp–bamboo/epoxy hybrid laminates: analysis and reproduction code

Analysis code for the manuscript **"Microcrystalline Cellulose Loading-Dependent Microstructure–Interface–Damage Relationships in Hemp–Bamboo/Epoxy Hybrid Laminates"** (Rakesh Kumar, G. Viranya, Meerja Akhil Jabbar, Yagna B. Adhyaru, D. Kishore, C. Ranjeeth Kumar, Davis Hans S J).

A single command regenerates every statistical result in the manuscript (Tables 9 and 10, the Tukey groupings in Figure 6, the loading regimes in Table 13), the data figures, the Figure 1 schematic, and a line-by-line numerical consistency audit of the Results and Discussion. The image-analysis, TGA/DSC and water-uptake routines that produced the microstructural, thermal and moisture data are included and validated against inputs of known ground truth.

```bash
pip install -r requirements.txt
python run_all.py          # about 25 s; everything is written to results/
python test_pipeline.py    # 12 verification tests
```

---

## 1. What is reproduced

| Manuscript item | Method | Code | Output in `results/` |
|---|---|---|---|
| Eq. 1, void fraction; Table 2 metrics | multi-Otsu segmentation, equivalent-circle diameter, nearest-neighbour fibre spacing | `image_analysis.py` | `microstructure_*.csv` |
| Eq. 2, water absorption; Fickian uptake | Crank plate solution, non-linear least squares | `moisture_retention.py` | `fickian_diffusion_fits.csv` |
| Eq. 3, property retention | ratio with first-order error propagation | `moisture_retention.py` | — |
| Table 11, TGA onset, T<sub>max</sub>, T<sub>g</sub> | ISO 11358-1 tangent onset, DTG peak, ISO 11357-2 half-step midpoint | `thermal_analysis.py` | `thermal_*.csv`, `thermal_conditions.json` |
| Tables 4–6, CV and 95 % CI | t-based interval, n = 10 | `stats_core.py` | `descriptives.csv` |
| Table 9, one-way ANOVA | F(df<sub>1</sub>, df<sub>2</sub>), p, partial η², ω², Welch F, Bartlett test | `stats_core.py`, `run_all.py` | `table9_anova.csv`, `table9_anova.md` |
| Figure 6, Tukey HSD | Tukey–Kramer comparisons, compact letter display | `stats_core.py`, `figures.py` | `tukey_pairwise.csv`, `tukey_letters.csv`, `fig6_tukey_tensile_strength.*` |
| Table 10, hybridization vs µCC | ANOVA over eight systems, % change, rule-of-mixtures hybrid effect | `run_all.py` | `control_groups.csv` |
| Section 4.7 / Table 8 | Pearson and Spearman with Fisher-z CI; standardised regression, VIF, LMG relative importance | `correlation_regression.py` | `correlations.csv`, `regression_tensile.csv` |
| Table 13, loading regimes | rule-based assignment from the Tukey groupings | `run_all.py` | `loading_regimes.csv` |
| Figures 3, 4 and 6; Figs S1–S3 | generated from the same Tukey output as the text | `figures.py` | `fig*.png` (600 dpi), `fig*.pdf` |
| Figure 1 | schematic drawn from geometric primitives | `figure1_schematic.py` | `figure1_schematic.*`, `figure1_legend.txt` |
| Internal consistency of the text | 152 automated checks | `consistency_audit.py` | `consistency_audit.md` |

---

## 2. Statistical design (Section 3.8)

µCC loading (0, 1, 2, 3 and 5 wt% of epoxy-resin mass) is the only experimental factor, so every loading effect is tested by **one-way ANOVA** followed by **Tukey's HSD**. No two-way or multifactor model is fitted anywhere in the pipeline.

For group sizes n<sub>i</sub>, means x̄<sub>i</sub>, standard deviations s<sub>i</sub> and k groups:

* SS<sub>between</sub> = Σ n<sub>i</sub>(x̄<sub>i</sub> − x̄)², SS<sub>within</sub> = Σ (n<sub>i</sub> − 1)s<sub>i</sub>²
* F = [SS<sub>between</sub>/(k − 1)] / [SS<sub>within</sub>/(N − k)], reported as F(k − 1, N − k)
* partial η² = SS<sub>between</sub>/(SS<sub>between</sub> + SS<sub>within</sub>) (equal to η² in a one-way design)
* ω² = (SS<sub>between</sub> − (k − 1)MS<sub>within</sub>)/(SS<sub>total</sub> + MS<sub>within</sub>)
* Tukey–Kramer: q<sub>ij</sub> = |x̄<sub>i</sub> − x̄<sub>j</sub>| / √[MS<sub>within</sub>/2 · (1/n<sub>i</sub> + 1/n<sub>j</sub>)], with p from the studentized range distribution

These quantities depend only on the group means, variances and sizes. ANOVA and Tukey results computed from the tabulated mean ± SD and n are therefore **identical** to those computed from the individual specimens. `test_pipeline.py` checks this against `scipy.stats.f_oneway` and `scipy.stats.tukey_hsd`.

Assumption checks run with every analysis. Bartlett's test uses summary statistics. Shapiro–Wilk and Brown–Forsythe run when specimen data are supplied. Welch's ANOVA and Games–Howell comparisons are reported alongside as variance-robust confirmations.

### Degrees of freedom used

| Response group | Specimens per formulation | Formulations | df |
|---|---|---|---|
| Tensile, flexural, impact (Tables 4–6) | 10 | 5 | (4, 45) |
| Microstructure and interface (Tables 2–3) | 5 | 5 | (4, 20) |
| Interrupted-load damage (Table 7) | 5 | 4 | (3, 16) |
| Thermal (Table 11) | 3 | 5 | (4, 10) |

The 95 % confidence intervals printed in Tables 4 and 5 are reproduced exactly with n = 10 (for example, 82 ± 4 MPa gives 79.1–84.9 MPa), which confirms the mechanical group size.

---

## 3. Data files

**`reported_data.csv`** contains every mean ± SD of Tables 2–7, 10, 11 and 12 and the 720 h values of Section 4.10, in long format (`table, response, unit, system, ucc_wt, load_level, mean, sd, n`). All results in `results/` are computed from this file.

**`specimen_data_template.csv`** gives the column layout for specimen-level measurements (one row per specimen, with microstructure measured on sections cut from the tested coupon). Save a filled copy as `specimen_data.csv`, or pass it with `--specimens`. Any response present in that file is then analysed from the individual specimens, with normality and variance tests added. The specimen-level regression of Table 8 is also run.

Raw instrument files go directly into the module CLIs:

```bash
python image_analysis.py   --images micrographs/   # L<wt>_S<specimen>_F<field>.tif (+ optional _fibre.png mask)
python thermal_analysis.py --curves thermal/       # TGA_L<wt>_R<rep>.csv, DSC_L<wt>_R<rep>.csv
python moisture_retention.py --uptake uptake.csv   # ucc_wt, specimen, time_h, mass_g
```

---

## 4. Procedures implemented

**Microstructure (Section 3.3).** Every micrograph receives the same magnification calibration (`pixel_size_um`), a median filter and a three-class multi-Otsu threshold. Voids are the dark class and µCC-rich features are the bright class. Features below a fixed pixel area are rejected. Field values are averaged per specimen, then per formulation. The run reports whether the target of at least 100 resolved µCC features per formulation was met. Fibre-spacing variation is the coefficient of variation of fibre nearest-neighbour distances. Resin-rich area is matrix lying farther than 15 µm from any fibre.

**Thermal analysis (Section 3.11).** TGA onset is the intersection of the pre-degradation baseline tangent with the tangent at the DTG inflection. T<sub>max</sub> is the DTG peak. T<sub>g</sub> is the half-step-height midpoint between extrapolated baselines on the second DSC heating scan. Heating rate, temperature range, atmosphere, purge flow and sample mass are set in `config.py` and written to `thermal_conditions.json` with each run.

**Moisture (Section 3.10).** Uptake follows Eq. 2 at 0, 24, 72, 168, 336, 504 and 720 h. The diffusion coefficient comes from the full Fickian series fit and is cross-checked with the initial-slope estimate.

**Relationships (Section 3.7).** Pearson correlation is used when both variables pass Shapiro–Wilk; otherwise Spearman. Both are reported with Fisher-z confidence intervals. Regression uses z-standardised variables. Each predictor's contribution is its LMG share of R², which is independent of entry order and sums exactly to R². VIF values are reported to show collinearity.

**Loading regimes (Section 3.12).** A loading is *effective reinforcement* when it shares the top Tukey group in at least two of tensile, flexural and impact strength. It is *excessive* when it lies above the optimum and at least one defect metric (void fraction, interfacial-gap index, agglomerate area fraction) is significantly higher than in the unfilled laminate. The rule recovers Table 13 from the data: 0–1 wt% low loading, 2–3 wt% effective reinforcement, 5 wt% excessive loading.

---

## 5. Key reproduced results

**One-way ANOVA, factor µCC loading (Table 9)**

| Response | F (df<sub>1</sub>, df<sub>2</sub>) | p | Partial η² |
|---|---|---|---|
| Tensile strength | 22.68 (4, 45) | < 0.001 | 0.67 |
| Tensile modulus | 12.46 (4, 45) | < 0.001 | 0.53 |
| Flexural strength | 23.11 (4, 45) | < 0.001 | 0.67 |
| Flexural modulus | 14.66 (4, 45) | < 0.001 | 0.57 |
| Impact strength | 26.88 (4, 45) | < 0.001 | 0.70 |
| Void fraction | 29.61 (4, 20) | < 0.001 | 0.86 |
| Interfacial-gap index | 21.23 (4, 20) | < 0.001 | 0.81 |
| Crack density, 90 % load | 29.30 (3, 16) | < 0.001 | 0.85 |

Bartlett's test does not reject equal variances for any of these responses (p > 0.70), and Welch's ANOVA gives the same conclusions.

**Tukey HSD groupings, tensile strength (Figure 6):** 0 wt% **d**, 1 wt% **c**, 2 wt% **ab**, 3 wt% **a**, 5 wt% **bc**. The 3 wt% laminate differs significantly from the unfilled laminate (Δ = +16 MPa, p < 0.001). The 1 and 5 wt% laminates fall in the same group (p = 0.98), and 2 and 3 wt% do not differ (p = 0.20).

---

## 6. Verification

`python test_pipeline.py` (or `pytest -q`) runs 12 tests:

* ANOVA, Tukey and Bartlett from summary statistics equal SciPy on specimen data built to the exact tabulated mean and SD
* the compact letter display gives the Figure 6 groupings
* the Table 4 confidence interval is reproduced
* image analysis recovers the void and agglomerate fractions of synthetic phantoms to within 0.15 and 0.25 percentage points
* TGA T<sub>max</sub> and DSC T<sub>g</sub> are recovered from analytic curves to within 1.0 °C and 0.5 °C
* the Fickian fit recovers M<sub>∞</sub> and D, and Eqs. 1–3 are checked numerically
* the LMG shares sum to R²

Each module also has its own `--selftest` option.

---

## 7. Repository files

| File | Purpose |
|---|---|
| `run_all.py` | single entry point; runs the full analysis and writes `results/` |
| `config.py` | all analysis settings: loadings, α, image parameters, TGA/DSC conditions |
| `reported_data.csv` | tabulated mean ± SD and n for every response |
| `specimen_data_template.csv` | column layout for specimen-level data |
| `stats_core.py` | descriptives, one-way ANOVA, Welch, Bartlett, Tukey–Kramer, Games–Howell, letter display |
| `consistency_audit.py` | cross-checks of text, tables, CV and CI columns, and ordinal statements |
| `image_analysis.py` | SEM micrograph quantification (Eq. 1) with phantom validation |
| `thermal_analysis.py` | TGA and DSC curve processing with analytic validation |
| `moisture_retention.py` | Eqs. 2–3 and Fickian diffusion |
| `correlation_regression.py` | correlation, standardised regression, LMG relative importance |
| `figures.py` | Figures 3, 4, 6 and S1–S3 |
| `figure1_schematic.py` | Figure 1 and its legend text |
| `test_pipeline.py` | verification tests |
| `requirements.txt` | Python dependencies |
| `CITATION.cff`, `.zenodo.json` | citation and archive metadata |

Requires Python ≥ 3.10. Verified with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, Matplotlib 3.11 and scikit-image 0.26.

## 8. Citation

Please cite the associated article together with the archived release of this repository (see `CITATION.cff`).
