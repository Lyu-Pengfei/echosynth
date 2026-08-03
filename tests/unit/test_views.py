"""View input contract, label mapping, and manifest round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from echosynth.data.manifest import write_manifest
from echosynth.views.view_classifier import MissingWeightsError, classify_views, prepare_view_input


def _row(path: Path) -> dict[str, object]:
    return {
        "image_id": "img-a",
        "patient_id": "patient-a",
        "encounter_id": "encounter-a",
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
        "split": "train",
    }


def test_view_mapping_confidence_and_round_trip(tmp_path: Path) -> None:
    image = tmp_path / "echo.png"
    Image.fromarray(np.full((112, 112), 80, dtype=np.uint8)).save(image)
    manifest = tmp_path / "manifest.csv"
    write_manifest(pd.DataFrame([_row(image)]), manifest)

    def model(batch: np.ndarray) -> np.ndarray:
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.8, 0.01]])

    result = classify_views(manifest, tmp_path / "unused.onnx", model=model)
    assert result.loc[0, "view_pred"] == "a3c"
    assert result.loc[0, "view_conf"] == pytest.approx(0.8)
    assert prepare_view_input(image).shape == (224, 224, 3)


def test_missing_weights_has_download_url(tmp_path: Path) -> None:
    with pytest.raises(MissingWeightsError, match="https://"):
        classify_views(tmp_path / "manifest.csv", tmp_path / "missing.onnx")
