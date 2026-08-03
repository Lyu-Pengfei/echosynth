"""Patient-split real-versus-synthetic discriminator (paper lines 286-293)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from echosynth.classify.engine import evaluate, train_classifier
from echosynth.classify.models import CompactCNN
from echosynth.classify.transforms import build_transform
from echosynth.data.datasets import EchoClassificationDataset
from echosynth.data.manifest import LeakageError
from echosynth.eval.delong import delong_ci
from echosynth.eval.guard import TestSetGuard
from echosynth.utils.checkpoint import load_checkpoint


@dataclass(frozen=True)
class DiscriminatorResult:
    """Test AUROC and DeLong interval for patient-separated discrimination."""

    auroc: float
    ci_low: float
    ci_high: float
    n_patients: int


def assert_patient_split(frame: pd.DataFrame) -> None:
    """Ensure no patient is shared across discriminator partitions."""
    required = {"patient_id", "split"}
    if not required.issubset(frame):
        raise ValueError(f"discriminator frame needs {sorted(required)}")
    counts = frame.groupby("patient_id", observed=True)["split"].nunique()
    offenders = sorted(counts[counts > 1].index.astype(str).tolist())
    if offenders:
        raise LeakageError(f"discriminator patient_id spans splits: {offenders[:20]}")


def make_patient_splits(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 2024,
    label_col: str = "is_real",
) -> pd.DataFrame:
    """Create deterministic label-stratified 70/15/15 patient partitions."""
    if not math.isclose(sum(ratios), 1) or any(value <= 0 for value in ratios):
        raise ValueError("patient split ratios must be positive and sum to one")
    patients = frame[["patient_id", label_col]].drop_duplicates()
    conflict = patients.groupby("patient_id")[label_col].nunique()
    bad = sorted(conflict[conflict > 1].index.astype(str).tolist())
    if bad:
        raise LeakageError(f"patient_id has conflicting real/synthetic labels: {bad[:20]}")
    generator = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for _, group in patients.groupby(label_col, sort=True, observed=True):
        identifiers = np.array(sorted(group["patient_id"].astype(str)), dtype=object)
        identifiers = identifiers[generator.permutation(len(identifiers))]
        train_count = math.floor(len(identifiers) * ratios[0])
        test_count = math.ceil(len(identifiers) * ratios[2])
        validation_count = len(identifiers) - train_count - test_count
        for identifier in identifiers[:train_count]:
            assignment[str(identifier)] = "train"
        for identifier in identifiers[train_count : train_count + validation_count]:
            assignment[str(identifier)] = "val"
        for identifier in identifiers[train_count + validation_count :]:
            assignment[str(identifier)] = "test"
    result = frame.copy()
    result["split"] = result["patient_id"].astype(str).map(assignment)
    assert_patient_split(result)
    return result


def discriminator_result(
    labels: np.ndarray, probabilities: np.ndarray, patient_count: int
) -> DiscriminatorResult:
    """Interpret AUROC 0.5 as no separability, alongside CI and partition composition."""
    auc, low, high = delong_ci(labels.astype(int).tolist(), probabilities.astype(float).tolist())
    return DiscriminatorResult(auc, low, high, patient_count)


def _loader(frame: pd.DataFrame, training: bool, batch_size: int) -> DataLoader[Any]:
    prepared = frame.copy()
    prepared["encounter_id"] = prepared["patient_id"].astype(str)
    prepared["_target"] = prepared["is_real"].astype(int).astype(str)
    dataset = EchoClassificationDataset(
        prepared, "_target", {"0": 0, "1": 1}, build_transform(training, 112)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=training)


def run_discriminator(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 16,
    seed: int = 2024,
) -> DiscriminatorResult:
    """Train on patient-separated data, select on val, and evaluate test once."""
    required = {"patient_id", "png_path", "is_real", "split"}
    if not required.issubset(frame):
        raise ValueError(f"discriminator frame missing columns: {sorted(required - set(frame))}")
    assert_patient_split(frame)
    train = frame.loc[frame["split"].astype(str) == "train"]
    validation = frame.loc[frame["split"].astype(str) == "val"]
    model = CompactCNN()
    training = train_classifier(
        model,
        _loader(train, True, batch_size),
        _loader(validation, False, batch_size),
        out_dir / "ckpt",
        epochs=epochs,
        patience=patience,
        seed=seed,
    )
    state, _ = load_checkpoint(training.checkpoint)
    model.load_state_dict(state["model"])
    test = frame.loc[frame["split"].astype(str) == "test"]
    evaluation = evaluate(
        model,
        _loader(test, False, batch_size),
        "test",
        TestSetGuard(out_dir / "test_access.jsonl"),
    )
    return discriminator_result(
        evaluation.encounters["label"].to_numpy(),
        evaluation.encounters["p"].to_numpy(),
        test["patient_id"].nunique(),
    )
