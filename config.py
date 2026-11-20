"""
Central configuration for the µCC hemp-bamboo/epoxy hybrid laminate analysis pipeline.

Every numerical setting used by the analysis is defined here, so that the
statistical, image-analysis and thermal procedures described in the
Methods section (Sections 3.3, 3.5, 3.7, 3.8, 3.10 and 3.11) are applied
identically to every formulation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTED_DATA = ROOT / "reported_data.csv"        # summary statistics of Tables 2-12
SPECIMEN_DATA = ROOT / "specimen_data.csv"        # optional specimen-level measurements
DEFAULT_OUTDIR = ROOT / "results"

# --------------------------------------------------------------------------
# Experimental design (Section 3.1)
# --------------------------------------------------------------------------
UCC_LOADINGS_WT = [0, 1, 2, 3, 5]                 # wt% of epoxy-resin mass
BASELINE_WT = 0                                   # unfilled hemp-bamboo/epoxy laminate
N_BATCHES = 3                                     # independent fabrication batches
HEMP_MASS_FRACTION_IN_HYBRID = 0.5                # hemp share of total fibre mass (hybrid)

CONTROL_SYSTEMS = [
    "neat epoxy",
    "hemp/epoxy",
    "bamboo/epoxy",
    "hemp-bamboo/epoxy",
    "hemp-bamboo + 1 wt% uCC",
    "hemp-bamboo + 2 wt% uCC",
    "hemp-bamboo + 3 wt% uCC",
    "hemp-bamboo + 5 wt% uCC",
]

# --------------------------------------------------------------------------
# Statistics (Section 3.8): one-way ANOVA with µCC loading as the only factor,
# followed by Tukey HSD (Tukey-Kramer form for unequal n).
# --------------------------------------------------------------------------
ALPHA = 0.05
CI_LEVEL = 0.95

# Responses reported in Table 9, in manuscript order.
# For crack density, the 90 % ultimate-load level of Table 7 is used.
TABLE9_RESPONSES = [
    ("tensile_strength", "", "Tensile strength"),
    ("tensile_modulus", "", "Tensile modulus"),
    ("flexural_strength", "", "Flexural strength"),
    ("flexural_modulus", "", "Flexural modulus"),
    ("impact_strength", "", "Impact strength"),
    ("void_fraction", "", "Void fraction"),
    ("interfacial_gap_index", "", "Interfacial-gap index"),
    ("crack_density", "90", "Crack density (90 % load)"),
]

MECHANICAL_RESPONSES = ["tensile_strength", "flexural_strength", "impact_strength"]
DEFECT_RESPONSES = ["void_fraction", "interfacial_gap_index", "agglomerate_area_fraction"]

# --------------------------------------------------------------------------
# Image analysis (Section 3.3, Eq. 1). Identical settings for all formulations.
# --------------------------------------------------------------------------
IMAGE = {
    "pixel_size_um": 0.25,          # µm per pixel; read from the SEM scale bar
    "databar_rows": 0,              # rows of SEM information bar to crop at the bottom
    "median_radius_px": 2,          # denoising before thresholding
    "n_classes": 3,                 # multi-Otsu classes: void / matrix+fibre / µCC-rich
    "min_void_area_px": 12,         # features smaller than this are treated as noise
    "min_agglomerate_area_px": 20,
    "resin_rich_distance_um": 15.0, # matrix farther than this from a fibre = resin-rich
    "min_features_per_formulation": 100,
}

# --------------------------------------------------------------------------
# Thermal analysis (Section 3.11). These conditions are written into every
# thermal result file so that the processing record matches Section 3.11.
# --------------------------------------------------------------------------
TGA_CONDITIONS = {
    "heating_rate_C_per_min": 10.0,
    "temperature_range_C": (30.0, 700.0),
    "atmosphere": "nitrogen",
    "purge_flow_mL_per_min": 50.0,
    "sample_mass_mg": (8.0, 10.0),
    "onset_method": "extrapolated tangent intersection (ISO 11358-1)",
}
DSC_CONDITIONS = {
    "heating_rate_C_per_min": 10.0,
    "temperature_range_C": (25.0, 200.0),
    "atmosphere": "nitrogen",
    "purge_flow_mL_per_min": 50.0,
    "sample_mass_mg": (5.0, 8.0),
    "programme": "heat-cool-heat; Tg taken from the second heating scan",
    "tg_method": "half-step-height midpoint (ISO 11357-2 / ASTM E1356)",
}

# --------------------------------------------------------------------------
# Moisture and ageing (Section 3.10, Eqs. 2-3)
# --------------------------------------------------------------------------
EXPOSURE_TIMES_H = [0, 24, 72, 168, 336, 504, 720]
SPECIMEN_THICKNESS_MM = 3.0         # used for the Fickian diffusion coefficient

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
FIG_DPI = 600
COLORS = {
    "series1": "#2a78d6",   # blue
    "series2": "#eb6834",   # orange
    "series3": "#1baf7a",   # aqua
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
}
