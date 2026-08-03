"""Known-pattern QC feature and manual-review tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from echosynth.data.manifest import write_manifest
from echosynth.data.qc import FEATURES, compute_features, run_qc


def _manifest_row(index: int, path: Path) -> dict[str, object]:
    return {
        "image_id": f"img-{index}",
        "patient_id": f"patient-{index}",
        "encounter_id": f"encounter-{index}",
        "group": "CA",
        "study_date": "2020-01-01",
        "dicom_path": f"/mounted/{index}.dcm",
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


def test_features_on_known_patterns() -> None:
    black = compute_features(np.zeros((32, 32), dtype=np.uint8))
    solid = compute_features(np.full((32, 32), 42, dtype=np.uint8))
    noise = compute_features(np.arange(256, dtype=np.uint8).reshape(16, 16))
    wedge = np.zeros((20, 20), dtype=np.uint8)
    wedge[:10] = 100
    covered = compute_features(wedge)
    assert tuple(black) == FEATURES
    assert black["nonblack_ratio"] == 0
    assert solid["intensity_std"] == 0
    assert solid["shannon_entropy"] == 0
    assert noise["shannon_entropy"] == pytest.approx(8.0)
    assert covered["sector_coverage"] == pytest.approx(0.5)


def test_flagged_is_retained_until_review_and_idempotent(tmp_path: Path) -> None:
    paths: list[Path] = []
    rows: list[dict[str, object]] = []
    for index in range(12):
        path = tmp_path / f"image-{index}.png"
        value = 255 if index == 11 else 10 + index
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(path)
        paths.append(path)
        rows.append(_manifest_row(index, path))
    manifest = tmp_path / "manifest.csv"
    queue = tmp_path / "queue.csv"
    write_manifest(pd.DataFrame(rows), manifest)
    first = run_qc(manifest, review_queue=queue)
    second = run_qc(manifest, review_queue=queue)
    pd.testing.assert_series_equal(first["qc_flag"], second["qc_flag"])
    assert first["qc_flag"].any()
    assert not first["qc_excluded"].any()
    reviewed = pd.read_csv(queue, keep_default_na=False)
    reviewed["reviewer_decision"] = "exclude"
    review_path = tmp_path / "reviewed.csv"
    reviewed.to_csv(review_path, index=False)
    applied = run_qc(manifest, review_queue=queue, reviewed=review_path)
    assert applied.loc[applied["qc_flag"].astype(bool), "qc_excluded"].all()
