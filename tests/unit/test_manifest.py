"""Manifest schema and leakage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from echosynth.data.manifest import LeakageError, read_manifest, write_manifest


def _row(image_id: str = "img-a") -> dict[str, object]:
    return {
        "image_id": image_id,
        "patient_id": "patient-a",
        "encounter_id": "encounter-a",
        "group": "CA",
        "study_date": "2020-01-01",
        "dicom_path": "/mounted/a.dcm",
        "png_path": "/derived/a.png",
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


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(pd.DataFrame([_row()]), path)
    loaded = read_manifest(path)
    assert loaded.loc[0, "image_id"] == "img-a"
    assert str(loaded.loc[0, "study_date"]) == "2020-01-01"


def test_encounter_spread_raises() -> None:
    first = _row("img-a")
    second = _row("img-b")
    second["png_path"] = "/derived/b.png"
    second["split"] = "test"
    with pytest.raises(LeakageError, match="encounter-a"):
        write_manifest(pd.DataFrame([first, second]), Path("unused.csv"))


def test_synthetic_test_row_raises(tmp_path: Path) -> None:
    row = _row()
    row.update({"task": "ca", "generator_run_id": "run-a", "split": "test"})
    with pytest.raises(LeakageError, match="img-a"):
        write_manifest(pd.DataFrame([row]), tmp_path / "unused.csv")
