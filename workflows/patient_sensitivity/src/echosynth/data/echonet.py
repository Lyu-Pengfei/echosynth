"""EchoNet-Dynamic and EchoNet-LVH manifest adapters (spec section 4.11)."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from echosynth.data.manifest import LeakageError
from echosynth.data.preprocess import preprocess_image
from echosynth.data.registration import CircleFit
from echosynth.utils.io import atomic_write_bytes, atomic_write_text
from echosynth.utils.logging import get_logger

LOGGER = get_logger(__name__)
DYNAMIC_EXPECTED = {"normal": 6961, "mild": 805, "severe": 1246}
LVH_EXPECTED = {
    "train": {0: 9152, 1: 1019},
    "val": {0: 1037, 1: 104},
    "test": {0: 210, 1: 129},
}


def lvef_class(ef: float) -> str:
    """Bucket LVEF as normal >=55, mild 40-<55, or severe <40."""
    if not np.isfinite(ef):
        raise ValueError("EF must be finite")
    if ef >= 55:
        return "normal"
    if ef >= 40:
        return "mild"
    return "severe"


def compute_lvh_threshold(training: pd.DataFrame) -> float:
    """Compute the 90th max-wall-thickness percentile from train rows only."""
    if "split" not in training or not training["split"].astype(str).eq("train").all():
        found = sorted(set(training.get("split", pd.Series(dtype=str)).astype(str)))
        raise LeakageError(f"LVH threshold input must contain train rows only; found {found}")
    required = {"ivsd", "lvpwd"}
    if not required.issubset(training):
        raise ValueError(f"LVH frame missing columns: {sorted(required - set(training))}")
    complete = training.dropna(subset=["ivsd", "lvpwd"])
    if complete.empty:
        raise ValueError("no complete training wall-thickness measurements")
    maximum = complete[["ivsd", "lvpwd"]].max(axis=1)
    return float(np.quantile(maximum.to_numpy(dtype=float), 0.9))


def adapt_lvh(
    frame: pd.DataFrame,
    *,
    validate_official: bool = False,
    official_threshold: float = 1.3274,
) -> tuple[pd.DataFrame, float]:
    """Drop incomplete measurements, derive train-only tau, and apply it unchanged."""
    required = {"video_id", "frame_path", "ivsd", "lvpwd", "split"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"EchoNet-LVH input missing columns: {sorted(missing)}")
    complete = frame.dropna(subset=["ivsd", "lvpwd"]).copy()
    dropped = len(frame) - len(complete)
    complete["max_wt"] = complete[["ivsd", "lvpwd"]].max(axis=1)
    threshold = compute_lvh_threshold(complete.loc[complete["split"].astype(str) == "train"])
    complete["wt_class"] = (complete["max_wt"] > threshold).astype(int)
    if validate_official:
        if dropped != 349 or len(complete) != 11651:
            raise ValueError(
                f"Official LVH completeness counts differ: dropped={dropped}, kept={len(complete)}"
            )
        if abs(threshold - official_threshold) > 1e-3:
            raise ValueError(
                f"Official train-only tau differs: {threshold} vs {official_threshold}"
            )
        counts = complete.groupby(["split", "wt_class"]).size().to_dict()
        expected = {
            (split, label): count
            for split, labels in LVH_EXPECTED.items()
            for label, count in labels.items()
        }
        if counts != expected:
            raise ValueError(f"Official LVH class counts differ: {counts}; expected {expected}")
    test = complete.loc[complete["split"].astype(str) == "test"]
    train = complete.loc[complete["split"].astype(str) == "train"]
    if not test.empty and not train.empty:
        test_fraction = float(test["wt_class"].mean())
        train_fraction = float(train["wt_class"].mean())
        if test_fraction > 0.3 and train_fraction < 0.15:
            warnings.warn(
                "LOUD: EchoNet-LVH official test minority prevalence is about 38% versus "
                "about 10% in train; AUPRC is not directly comparable across partitions.",
                RuntimeWarning,
                stacklevel=2,
            )
    return complete[
        ["video_id", "frame_path", "ivsd", "lvpwd", "max_wt", "wt_class", "split"]
    ], threshold


def adapt_dynamic(frame: pd.DataFrame, *, validate_expected: bool = False) -> pd.DataFrame:
    """Apply the predefined three-class LVEF buckets and log count deviations."""
    required = {"video_id", "frame_path", "ef", "split"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"EchoNet-Dynamic input missing columns: {sorted(missing)}")
    result = frame.copy()
    result["lvef_class"] = result["ef"].astype(float).map(lvef_class)
    counts = result["lvef_class"].value_counts().to_dict()
    LOGGER.info("EchoNet-Dynamic LVEF counts: %s", counts)
    deviations = {
        label: abs(int(counts.get(label, 0)) - expected) / expected
        for label, expected in DYNAMIC_EXPECTED.items()
    }
    if any(value > 0.01 for value in deviations.values()):
        warnings.warn(
            f"EchoNet-Dynamic class counts deviate from expected by >1%: {counts}",
            RuntimeWarning,
            stacklevel=2,
        )
    if validate_expected and any(value > 0.01 for value in deviations.values()):
        raise ValueError(f"Official Dynamic counts differ: {counts}")
    return result[["video_id", "frame_path", "ef", "lvef_class", "split"]]


def decode_first_video_frame(video: Path) -> np.ndarray:
    """Decode frame index zero using the same predefined rule as DICOM ingest."""
    capture = cv2.VideoCapture(str(video))
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        raise ValueError(f"could not decode first frame: {video}")
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cast(np.ndarray, frame)


def ingest_dynamic_videos(
    metadata: pd.DataFrame,
    video_root: Path,
    out_dir: Path,
    reference: CircleFit,
    manifest_out: Path,
) -> pd.DataFrame:
    """Decode first frames and reuse the S3 preprocessing path without forked logic."""
    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        video_id = str(row["video_id"])
        processed = preprocess_image(decode_first_video_frame(video_root / video_id), reference)
        destination = out_dir / f"{Path(video_id).stem}.png"
        buffer = io.BytesIO()
        Image.fromarray(processed, mode="L").save(buffer, "PNG")
        atomic_write_bytes(destination, buffer.getvalue())
        rows.append(
            {
                "video_id": video_id,
                "frame_path": str(destination),
                "ef": float(row["ef"]),
                "split": str(row["split"]),
            }
        )
    result = adapt_dynamic(pd.DataFrame(rows))
    atomic_write_text(manifest_out, result.to_csv(index=False))
    return result
