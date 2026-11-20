"""
Quantitative microstructure analysis of polished cross-section micrographs
(Section 3.3, Eq. 1).

For each SEM / optical field the script measures
  * void fraction            V_v = A_v / A_t x 100                     (Eq. 1)
  * µCC agglomerate size     mean equivalent-circle diameter (µm)
  * agglomerate area fraction (%) and number density (mm^-2)
  * fibre-spacing variation  CV of fibre nearest-neighbour distances (%)
  * resin-rich area          matrix farther than a set distance from any fibre (%)

All fields are processed with the same magnification calibration, the same
denoising and the same multi-Otsu thresholding, so that differences between
formulations reflect the material and not the processing.

File naming for batch mode:  L<wt>_S<specimen>_F<field>.<tif|png|jpg>
An optional fibre mask with the same stem and suffix "_fibre.png" enables the
fibre-spacing and resin-rich-area measurements.

Usage
  python image_analysis.py --selftest
  python image_analysis.py --images path/to/micrographs --out results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, io, measure, morphology, util

import config

NAME_RE = re.compile(r"L(?P<wt>\d+(?:\.\d+)?)_S(?P<spec>\d+)_F(?P<field>\d+)", re.I)


# --------------------------------------------------------------------------
# Loading and segmentation
# --------------------------------------------------------------------------
def load_image(path: str | Path, databar_rows: int = 0) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = util.img_as_float(img.astype(np.float64) / (img.max() or 1))
    if databar_rows:
        img = img[:-databar_rows]
    return img


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected features smaller than min_px pixels (version-independent)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def segment(img: np.ndarray, p: dict = config.IMAGE) -> dict:
    """Return void and µCC-rich masks from a fixed multi-Otsu procedure."""
    smooth = filters.median(img, morphology.disk(p["median_radius_px"]))
    t_low, t_high = filters.threshold_multiotsu(smooth, classes=p["n_classes"])[[0, -1]]
    voids = _drop_small(smooth < t_low, p["min_void_area_px"])
    aggl = _drop_small(smooth > t_high, p["min_agglomerate_area_px"])
    return {"void": voids, "agglomerate": aggl, "thresholds": (float(t_low), float(t_high))}


# --------------------------------------------------------------------------
# Field metrics
# --------------------------------------------------------------------------
def void_fraction(void_mask: np.ndarray) -> float:
    """Eq. 1: V_v = A_v / A_t x 100."""
    return 100.0 * void_mask.sum() / void_mask.size


def agglomerate_metrics(mask: np.ndarray, px_um: float) -> dict:
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    area_mm2 = mask.size * (px_um * 1e-3) ** 2
    ecd = np.array([p.equivalent_diameter_area for p in props]) * px_um
    return {"agglomerate_count": len(props),
            "agglomerate_size_um": float(ecd.mean()) if ecd.size else np.nan,
            "agglomerate_area_fraction_pct": 100.0 * mask.sum() / mask.size,
            "agglomerate_number_density_per_mm2": len(props) / area_mm2}


def fibre_metrics(fibre_mask: np.ndarray, px_um: float, resin_rich_um: float) -> dict:
    lab = measure.label(fibre_mask)
    cent = np.array([p.centroid for p in measure.regionprops(lab)])
    if len(cent) < 3:
        return {"fiber_spacing_variation_pct": np.nan, "resin_rich_area_pct": np.nan}
    d, _ = cKDTree(cent).query(cent, k=2)
    nnd = d[:, 1] * px_um
    dist = ndi.distance_transform_edt(~fibre_mask) * px_um
    matrix = ~fibre_mask
    rr = (dist > resin_rich_um) & matrix
    return {"fiber_spacing_variation_pct": 100.0 * nnd.std(ddof=1) / nnd.mean(),
            "resin_rich_area_pct": 100.0 * rr.sum() / fibre_mask.size}


def quantify_field(img: np.ndarray, fibre_mask: np.ndarray | None = None,
                   p: dict = config.IMAGE) -> dict:
    seg = segment(img, p)
    out = {"void_fraction_pct": float(void_fraction(seg["void"])),
           "threshold_low": seg["thresholds"][0], "threshold_high": seg["thresholds"][1]}
    out.update(agglomerate_metrics(seg["agglomerate"], p["pixel_size_um"]))
    if fibre_mask is not None:
        out.update(fibre_metrics(fibre_mask.astype(bool), p["pixel_size_um"],
                                 p["resin_rich_distance_um"]))
    return out


# --------------------------------------------------------------------------
# Batch processing: field -> specimen -> formulation
# --------------------------------------------------------------------------
def analyse_directory(folder: str | Path, p: dict = config.IMAGE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        if f.stem.endswith("_fibre"):
            continue
        m = NAME_RE.search(f.stem)
        if not m:
            continue
        img = load_image(f, p["databar_rows"])
        fm = f.with_name(f.stem + "_fibre.png")
        fibre = load_image(fm, p["databar_rows"]) > 0.5 if fm.exists() else None
        r = quantify_field(img, fibre, p)
        r.update(file=f.name, ucc_wt=float(m["wt"]), specimen=int(m["spec"]), field=int(m["field"]))
        rows.append(r)
    fields = pd.DataFrame(rows)
    if fields.empty:
        raise FileNotFoundError(f"No micrographs matching L<wt>_S<n>_F<n> found in {folder}")
    metric_cols = [c for c in fields.columns if c.endswith(("_pct", "_um", "_per_mm2"))]
    spec = fields.groupby(["ucc_wt", "specimen"])[metric_cols].mean().reset_index()
    form = spec.groupby("ucc_wt")[metric_cols].agg(["mean", "std", "count"])
    form.columns = ["_".join(c) for c in form.columns]
    counts = fields.groupby("ucc_wt")["agglomerate_count"].sum()
    form["features_measured"] = counts
    form["feature_target_met"] = counts >= p["min_features_per_formulation"]
    return fields, form.reset_index()


# --------------------------------------------------------------------------
# Validation phantom with known ground truth
# --------------------------------------------------------------------------
def make_phantom(size=1024, n_voids=60, void_r=(3, 9), n_aggl=45, aggl_r=(4, 14),
                 fibre_r=22, fibre_pitch=70, jitter=12, noise=0.035, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.full((size, size), 0.50)
    fibre = np.zeros((size, size), bool)
    for cy in range(fibre_pitch // 2, size, fibre_pitch):
        for cx in range(fibre_pitch // 2, size, fibre_pitch):
            y, x = cy + rng.integers(-jitter, jitter + 1), cx + rng.integers(-jitter, jitter + 1)
            fibre |= (yy - y) ** 2 + (xx - x) ** 2 <= fibre_r ** 2
    img[fibre] = 0.56

    def blobs(n, rr, val):
        m = np.zeros((size, size), bool)
        for _ in range(n):
            r = rng.uniform(*rr)
            y, x = rng.uniform(r, size - r, 2)
            m |= (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
        img[m] = val
        return m

    void = blobs(n_voids, void_r, 0.08)
    aggl = blobs(n_aggl, aggl_r, 0.92) & ~void
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
    truth = {"void_fraction_pct": void_fraction(void),
             "agglomerate_area_fraction_pct": 100 * aggl.sum() / aggl.size}
    return img, fibre, truth


def selftest(verbose=True) -> bool:
    ok = True
    for seed in range(3):
        img, fibre, truth = make_phantom(seed=seed)
        r = quantify_field(img, fibre)
        e1 = abs(r["void_fraction_pct"] - truth["void_fraction_pct"])
        e2 = abs(r["agglomerate_area_fraction_pct"] - truth["agglomerate_area_fraction_pct"])
        good = e1 < 0.15 and e2 < 0.25
        ok &= good
        if verbose:
            print(f"phantom {seed}: void {r['void_fraction_pct']:.3f} % (true "
                  f"{truth['void_fraction_pct']:.3f}), agglomerate area "
                  f"{r['agglomerate_area_fraction_pct']:.3f} % (true "
                  f"{truth['agglomerate_area_fraction_pct']:.3f}) -> {'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of micrographs named L<wt>_S<n>_F<n>")
    ap.add_argument("--out", default=str(config.DEFAULT_OUTDIR))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.images:
        raise SystemExit(0 if selftest() else 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fields, form = analyse_directory(a.images)
    fields.to_csv(out / "microstructure_fields.csv", index=False)
    form.to_csv(out / "microstructure_by_formulation.csv", index=False)
    print(form.to_string(index=False))


if __name__ == "__main__":
    main()
