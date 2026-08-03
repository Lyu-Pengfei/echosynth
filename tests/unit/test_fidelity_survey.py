"""Fidelity, entropy, denominator, and discriminator partition tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from echosynth.data.manifest import LeakageError
from echosynth.eval.discriminator import assert_patient_split, make_patient_splits
from echosynth.eval.fidelity import histogram_metrics
from echosynth.eval.survey import analyze_survey, normalized_entropy


def _distribution(values: list[float]) -> np.ndarray:
    result = np.full(256, 1e-12)
    result[: len(values)] = values
    return result / result.sum()


def test_fidelity_identities_and_hand_example() -> None:
    distribution = _distribution([0.1, 0.2, 0.3, 0.4])
    identical = histogram_metrics(distribution, distribution)
    assert identical.kl_real_syn == pytest.approx(0, abs=1e-15)
    assert identical.js_divergence == pytest.approx(0, abs=1e-15)
    first = _distribution([0.4, 0.3, 0.2, 0.1])
    second = _distribution([0.1, 0.2, 0.3, 0.4])
    metrics = histogram_metrics(first, second)
    expected_kl = sum(p * math.log(p / q) for p, q in zip(first, second, strict=True))
    assert metrics.kl_real_syn == pytest.approx(expected_kl)
    assert metrics.js_distance == pytest.approx(math.sqrt(metrics.js_divergence))
    assert metrics.rmse_dist == pytest.approx(math.sqrt(metrics.brier_dist))


def test_entropy_and_item_wise_denominators() -> None:
    assert normalized_entropy(["a", "b", "a", "b"], ["a", "b"]) == pytest.approx(1)
    assert normalized_entropy(["a", "a"], ["a", "b"]) == pytest.approx(0)
    frame = pd.DataFrame(
        {
            "source": ["real", "real", "real"],
            "usable": ["yes", "yes", None],
            "roi": ["yes", "no", "yes"],
        }
    )
    table = analyze_survey(frame, ["usable", "roi"])
    assert set(table.loc[table["item"] == "usable", "n"]) == {2}
    assert set(table.loc[table["item"] == "roi", "n"]) == {3}
    with pytest.raises(NotImplementedError, match="descriptive"):
        analyze_survey(frame, ["usable"], confirmatory_test=True)


def test_discriminator_patient_partition() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(20)],
            "is_real": [index < 10 for index in range(20)],
        }
    )
    split = make_patient_splits(frame)
    assert_patient_split(split)
    bad = pd.concat([split, split.iloc[[0]].assign(split="test")], ignore_index=True)
    with pytest.raises(LeakageError, match=str(split.loc[0, "patient_id"])):
        assert_patient_split(bad)
