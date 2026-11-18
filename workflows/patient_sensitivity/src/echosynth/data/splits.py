"""Deterministic encounter-level stratified splitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from echosynth.data.manifest import LeakageError, assert_manifest_invariants


def make_encounter_splits(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 2024,
    stratify_by: str = "group",
) -> pd.DataFrame:
    """Assign every encounter to one stratified split (spec section 3.4)."""
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("ratios must be three positive values summing to one")
    required = {"encounter_id", stratify_by}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")
    encounter_labels = df[["encounter_id", stratify_by]].drop_duplicates()
    conflicting = encounter_labels.groupby("encounter_id")[stratify_by].nunique()
    bad = sorted(conflicting[conflicting > 1].index.astype(str).tolist())
    if bad:
        raise LeakageError(f"encounter_id has inconsistent {stratify_by}: {bad[:20]}")
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for _, group in encounter_labels.groupby(stratify_by, sort=True, observed=True):
        ids = np.array(sorted(group["encounter_id"].astype(str).tolist()), dtype=object)
        ids = ids[rng.permutation(len(ids))]
        n_train = math.floor(len(ids) * ratios[0])
        n_test = math.ceil(len(ids) * ratios[2])
        n_val = len(ids) - n_train - n_test
        for encounter_id in ids[:n_train]:
            assignment[str(encounter_id)] = "train"
        for encounter_id in ids[n_train : n_train + n_val]:
            assignment[str(encounter_id)] = "val"
        for encounter_id in ids[n_train + n_val :]:
            assignment[str(encounter_id)] = "test"
    result = df.copy()
    result["split"] = result["encounter_id"].astype(str).map(assignment)
    assert_manifest_invariants(result)
    return result
