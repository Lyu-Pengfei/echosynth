"""Load-bearing metric, aggregation, and DeLong validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from echosynth.eval.aggregate import to_encounter_level
from echosynth.eval.delong import (
    delong_auc_variance,
    delong_paired_delta_ci,
    normal_delta_ci,
)
from echosynth.eval.metrics import auprc_ci, auroc


def test_basic_auc_and_auprc_policy() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1
    assert auroc([0, 0, 1, 1], [0.5] * 4) == 0.5
    with pytest.raises(NotImplementedError, match="line 239"):
        auprc_ci([0, 1], [0.2, 0.8])


def test_delong_matches_sklearn_on_random_vectors() -> None:
    generator = np.random.default_rng(8)
    labels = np.array([0] * 20 + [1] * 20)
    for _ in range(10):
        predictions = generator.random(40)
        auc, variance = delong_auc_variance(labels, predictions)
        assert auc == pytest.approx(roc_auc_score(labels, predictions), abs=1e-10)
        assert variance >= 0


def test_delong_fixed_variance_and_identical_pair() -> None:
    labels = [1, 1, 0, 0]
    predictions = [0.9, 0.6, 0.7, 0.1]
    auc, variance = delong_auc_variance(labels, predictions)
    assert auc == pytest.approx(0.75)
    assert variance == pytest.approx(0.125)
    paired = delong_paired_delta_ci(labels, predictions, predictions)
    assert paired.delta == 0
    assert paired.p_value == 1


def test_reported_delta_interval_construction() -> None:
    low, high = normal_delta_ci(0.041, 0.0143)
    assert low == pytest.approx(0.013, abs=5e-4)
    assert high == pytest.approx(0.069, abs=5e-4)


def test_encounter_aggregation_and_consistency() -> None:
    frame = pd.DataFrame(
        {"encounter_id": ["a", "a", "a"], "label": [1, 1, 1], "p": [0.2, 0.4, 0.9]}
    )
    result = to_encounter_level(frame)
    assert result.loc[0, "p"] == pytest.approx(0.5)
    bad = frame.copy()
    bad.loc[2, "label"] = 0
    with pytest.raises(ValueError, match="inconsistent label"):
        to_encounter_level(bad)
