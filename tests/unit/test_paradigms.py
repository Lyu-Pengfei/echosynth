"""Validation selection and matched-bootstrap tests."""

from __future__ import annotations

import pandas as pd

from echosynth.classify.paradigms import augment_training_data, select_synthetic_count


def test_bootstrap_matches_synthetic_arm_count() -> None:
    real = pd.DataFrame(
        {
            "image_id": ["r1", "r2", "r3"],
            "group": ["CA", "GEN", "GEN"],
            "split": ["train"] * 3,
        }
    )
    synthetic = pd.DataFrame(
        {
            "image_id": [f"s{index}" for index in range(5)],
            "class_label": ["CA"] * 5,
            "split": ["train"] * 5,
        }
    )
    trstr = augment_training_data(real, synthetic, "TRSTR", 4)
    bootstrap = augment_training_data(real, synthetic, "BOOTSTRAP", 4)
    assert len(trstr) - len(real) == 4
    assert len(bootstrap) - len(real) == 4
    assert (bootstrap.iloc[len(real) :]["group"] == "CA").all()


def test_selection_uses_only_validation_scorer() -> None:
    calls: list[int] = []

    def scorer(count: int) -> float:
        calls.append(count)
        return {100: 0.7, 200: 0.8}[count]

    selected, curve = select_synthetic_count([100, 200], scorer)
    assert selected == 200
    assert calls == [100, 200]
    assert [point.val_auroc for point in curve] == [0.7, 0.8]
