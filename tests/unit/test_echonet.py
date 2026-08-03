"""EchoNet label thresholds, missingness, and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from echosynth.data.echonet import adapt_dynamic, adapt_lvh, compute_lvh_threshold, lvef_class
from echosynth.data.manifest import LeakageError


def test_lvef_boundaries() -> None:
    assert lvef_class(55) == "normal"
    assert lvef_class(40) == "mild"
    assert lvef_class(39.99) == "severe"
    result = adapt_dynamic(
        pd.DataFrame(
            {
                "video_id": ["a", "b", "c"],
                "frame_path": ["a.png", "b.png", "c.png"],
                "ef": [60, 45, 30],
                "split": ["train", "val", "test"],
            }
        )
    )
    assert result["lvef_class"].tolist() == ["normal", "mild", "severe"]


def test_lvh_threshold_and_missing_row() -> None:
    values = np.linspace(0.8, 1.5, 50)
    frame = pd.DataFrame(
        {
            "video_id": [f"v{index}" for index in range(50)],
            "frame_path": [f"v{index}.png" for index in range(50)],
            "ivsd": values,
            "lvpwd": values - 0.1,
            "split": ["train"] * 40 + ["val"] * 5 + ["test"] * 5,
        }
    )
    frame.loc[49, "lvpwd"] = np.nan
    expected_tau = float(np.quantile(values[:40], 0.9))
    result, threshold = adapt_lvh(frame)
    assert threshold == pytest.approx(expected_tau)
    assert len(result) == 49
    assert (result["wt_class"] == (result["max_wt"] > threshold).astype(int)).all()


def test_tau_rejects_nontraining_rows() -> None:
    frame = pd.DataFrame({"ivsd": [1.0, 1.2], "lvpwd": [1.0, 1.1], "split": ["train", "val"]})
    with pytest.raises(LeakageError, match="train rows only"):
        compute_lvh_threshold(frame)
