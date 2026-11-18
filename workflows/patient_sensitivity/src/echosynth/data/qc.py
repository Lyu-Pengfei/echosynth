"""Eight-feature high-sensitivity image quality screen (paper lines 147-165)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kurtosis, skew

from echosynth.data.manifest import read_manifest, write_manifest

FEATURES = (
    "sector_coverage",
    "nonblack_ratio",
    "intensity_mean",
    "intensity_std",
    "shannon_entropy",
    "skewness",
    "kurtosis",
    "edge_density",
)


def _finite_stat(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def compute_features(image: np.ndarray, canny: tuple[int, int] = (50, 150)) -> dict[str, float]:
    """Compute the ordered QC screen from paper section 2.3, lines 147-165.

    Entropy is base-2 over a 256-bin histogram. Kurtosis uses SciPy's Fisher definition.
    Constant frames have defined skewness and kurtosis of zero rather than propagating NaN.
    """
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"QC expects a grayscale 2-D frame, got {array.shape}")
    if array.size == 0:
        raise ValueError("QC cannot process an empty image")
    pixels = np.clip(array, 0, 255).astype(np.uint8)
    nonzero = pixels > 0
    histogram = np.histogram(pixels, bins=256, range=(0, 256))[0].astype(np.float64)
    probabilities = histogram[histogram > 0] / pixels.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    flattened = pixels.astype(np.float64).ravel()
    edges = cv2.Canny(pixels, int(canny[0]), int(canny[1]))
    constant = bool(np.all(flattened == flattened[0]))
    values = (
        float(nonzero.mean()),
        float(nonzero.mean()),
        float(flattened.mean()),
        float(flattened.std(ddof=0)),
        entropy,
        0.0 if constant else _finite_stat(float(skew(flattened, bias=True))),
        0.0 if constant else _finite_stat(float(kurtosis(flattened, fisher=True, bias=True))),
        float((edges > 0).mean()),
    )
    return dict(zip(FEATURES, values, strict=True))


def _standardize(features: pd.DataFrame) -> pd.DataFrame:
    means = features.mean(axis=0)
    standard_deviations = features.std(axis=0, ddof=0).replace(0, np.nan)
    return ((features - means) / standard_deviations).fillna(0.0)


def apply_review(frame: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit human decisions; unconfirmed flags remain retained."""
    required = {"image_id", "reviewer_decision"}
    if not required.issubset(reviewed):
        raise ValueError(f"review CSV requires columns {sorted(required)}")
    decisions = reviewed.set_index("image_id")["reviewer_decision"]
    flagged_ids = set(frame.loc[frame["qc_flag"].astype(bool), "image_id"].astype(str))
    missing = sorted(
        image_id
        for image_id in flagged_ids
        if image_id not in decisions.index or not str(decisions.loc[image_id]).strip()
    )
    if missing:
        raise ValueError(f"Flagged images require non-empty reviewer_decision: {missing[:20]}")
    accepted = {"keep", "retain", "include", "accepted"}
    excluded = {"exclude", "reject", "excluded", "rejected"}
    result = frame.copy()
    result["qc_excluded"] = False
    for index, row in result.iterrows():
        image_id = str(row["image_id"])
        if not bool(row["qc_flag"]):
            continue
        decision = str(decisions.loc[image_id]).strip().lower()
        if decision not in accepted | excluded:
            raise ValueError(f"Unknown reviewer_decision for {image_id}: {decision}")
        result.at[index, "qc_excluded"] = decision in excluded
        result.at[index, "qc_reason"] = f"manual_review:{decision}"
    return result


def run_qc(
    manifest: Path,
    *,
    canny: tuple[int, int] = (50, 150),
    review_queue: Path = Path("manifests/qc_review_queue.csv"),
    reviewed: Path | None = None,
) -> pd.DataFrame:
    """Compute z-score flags and update a manifest idempotently.

    A flag is only a high-sensitivity screen. It never sets ``qc_excluded`` without a
    complete human review file, matching paper lines 159-165.
    """
    frame = read_manifest(manifest)
    raw_rows: list[dict[str, float]] = []
    for path in frame["png_path"].astype(str):
        with Image.open(path) as image:
            raw_rows.append(compute_features(np.asarray(image.convert("L")), canny))
    raw = pd.DataFrame(raw_rows, columns=FEATURES, index=frame.index)
    zscores = _standardize(raw)
    frame["qc_flag"] = (zscores.abs() > 3).any(axis=1)
    frame["qc_features"] = [
        json.dumps(row, sort_keys=True) for row in raw.to_dict(orient="records")
    ]
    frame["qc_excluded"] = False
    queue = pd.concat(
        [
            frame[["image_id", "png_path", "qc_flag"]],
            raw,
            zscores.add_suffix("_z"),
        ],
        axis=1,
    )
    queue = queue.loc[queue["qc_flag"].astype(bool)].copy()
    queue["reviewer_decision"] = ""
    review_queue.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(review_queue, index=False)
    if reviewed is not None:
        frame = apply_review(frame, pd.read_csv(reviewed, keep_default_na=False))
    write_manifest(frame, manifest)
    return frame
