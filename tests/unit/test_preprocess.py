"""Synthetic-wedge registration, masking, and leakage tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from echosynth.data.manifest import LeakageError, write_manifest
from echosynth.data.masking import sector_mask
from echosynth.data.preprocess import preprocess_image
from echosynth.data.registration import (
    CircleFit,
    build_reference,
    detect_sector,
    fit_circular_model,
)


def _wedge(
    shape: tuple[int, int] = (160, 160),
    center: tuple[float, float] = (80, 20),
    radius: float = 115,
    angle: float = math.pi / 2,
    opening: float = math.radians(65),
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    dx = x - center[0]
    dy = y - center[1]
    radial = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    delta = np.angle(np.exp(1j * (theta - angle)))
    mask = (radial <= radius) & (np.abs(delta) <= opening / 2)
    image = np.zeros(shape, dtype=np.uint8)
    image[mask] = np.clip(180 - radial[mask], 30, 180).astype(np.uint8)
    return image


def _row(path: Path, split: str = "train") -> dict[str, object]:
    return {
        "image_id": f"img-{split}",
        "patient_id": f"patient-{split}",
        "encounter_id": f"encounter-{split}",
        "group": "CA",
        "study_date": "2020-01-01",
        "dicom_path": "/mounted/source.dcm",
        "png_path": str(path),
        "sop_class_uid": "1.2.3",
        "color_doppler": False,
        "view_pred": None,
        "view_conf": None,
        "qc_flag": False,
        "qc_features": json.dumps({}),
        "qc_excluded": False,
        "qc_reason": "",
        "split": split,
    }


def test_detection_alignment_mask_and_output_contract() -> None:
    image = _wedge(center=(72, 25), angle=math.radians(82))
    sector = detect_sector(image)
    assert sector is not None
    fit = fit_circular_model(sector)
    reference = CircleFit(80, 18, 110, math.pi / 2, math.radians(65), 0)
    output = preprocess_image(image, reference)
    assert output.shape == (112, 112)
    assert output.dtype == np.uint8
    masked = sector_mask(image, fit)
    assert np.array_equal(masked, sector_mask(masked, fit))
    assert output[0, 0] == 0


def test_build_reference_rejects_validation_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "wedge.png"
    Image.fromarray(_wedge()).save(image_path)
    manifest = tmp_path / "manifest.csv"
    rows = [_row(image_path, "train"), _row(image_path, "val")]
    rows[1]["image_id"] = "img-val"
    rows[1]["png_path"] = str(tmp_path / "other.png")
    Image.fromarray(_wedge()).save(rows[1]["png_path"])
    write_manifest(pd.DataFrame(rows), manifest)
    with pytest.raises(LeakageError, match="train only"):
        build_reference(manifest, tmp_path / "reference.json")
