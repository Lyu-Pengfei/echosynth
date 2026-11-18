"""Shared, leakage-safe utilities for the reviewer follow-up analyses."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SEEDS = (2024, 2025, 2026, 2027, 2028)
PARADIGMS = ("TRTR", "TRSTR")


def run_dirs(v6_root: Path) -> list[Path]:
    return [v6_root / "results/full_revision/runs" / f"ca_revision_full_seed{s}" for s in SEEDS]


def ensemble_predictions(v6_root: Path, filename: str) -> pd.DataFrame:
    """Average the five fixed-seed encounter probabilities after exact key checks."""
    keys = ["encounter_id", "patient_id", "group", "split", "label"]
    frames: list[pd.DataFrame] = []
    for seed, directory in zip(SEEDS, run_dirs(v6_root), strict=True):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)[keys + ["p"]].rename(columns={"p": f"p_{seed}"})
        if frame["encounter_id"].duplicated().any():
            raise RuntimeError(f"duplicate encounters in {path}")
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(frames[0]):
        raise RuntimeError("seed prediction files do not contain identical encounters")
    merged["p"] = merged[[f"p_{seed}" for seed in SEEDS]].mean(axis=1)
    return merged[keys + ["p"]]


def ensemble_from_dirs(directories: Iterable[Path], filename: str) -> pd.DataFrame:
    keys = ["encounter_id", "patient_id", "group", "split", "label"]
    frames: list[pd.DataFrame] = []
    for index, directory in enumerate(directories):
        frame = pd.read_csv(directory / filename)[keys + ["p"]]
        frames.append(frame.rename(columns={"p": f"p_seed{index}"}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    probability_columns = [column for column in merged if column.startswith("p_seed")]
    merged["p"] = merged[probability_columns].mean(axis=1)
    return merged[keys + ["p"]]


def patient_level(frame: pd.DataFrame) -> pd.DataFrame:
    consistency = frame.groupby("patient_id")["label"].nunique()
    if (consistency > 1).any():
        raise RuntimeError("conflicting labels within patient")
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        p=("p", "mean"),
        group=("group", "first"),
        split=("split", "first"),
        n_encounters=("encounter_id", "nunique"),
    )


def point_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["p"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def stratified_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int,
    replicates: int = 2000,
) -> tuple[float, float, int]:
    """Bootstrap independent positive and negative patients with replacement."""
    positive = frame.loc[frame["label"].astype(int) == 1]
    negative = frame.loc[frame["label"].astype(int) == 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan"), float("nan"), 0
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = pd.concat(
            [
                positive.iloc[generator.integers(0, len(positive), len(positive))],
                negative.iloc[generator.integers(0, len(negative), len(negative))],
            ],
            ignore_index=True,
        )
        try:
            value = float(statistic(sample))
        except (ValueError, FloatingPointError, RuntimeError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(100, replicates // 2):
        return float("nan"), float("nan"), len(values)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def auc_ci(frame: pd.DataFrame, seed: int, replicates: int = 2000) -> tuple[float, float, float]:
    point = float(roc_auc_score(frame["label"], frame["p"]))
    low, high, _ = stratified_bootstrap(
        frame,
        lambda x: roc_auc_score(x["label"], x["p"]),
        seed=seed,
        replicates=replicates,
    )
    return point, low, high


def paired_delta_ci(
    real: pd.DataFrame,
    augmented: pd.DataFrame,
    *,
    seed: int,
    replicates: int = 2000,
) -> tuple[float, float, float]:
    merged = real[["patient_id", "label", "p"]].merge(
        augmented[["patient_id", "p"]], on="patient_id", suffixes=("_real", "_aug")
    )
    positive = merged.loc[merged["label"] == 1]
    negative = merged.loc[merged["label"] == 0]
    point = float(
        roc_auc_score(merged["label"], merged["p_aug"])
        - roc_auc_score(merged["label"], merged["p_real"])
    )
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = pd.concat(
            [
                positive.iloc[generator.integers(0, len(positive), len(positive))],
                negative.iloc[generator.integers(0, len(negative), len(negative))],
            ]
        )
        values.append(
            float(
                roc_auc_score(sample["label"], sample["p_aug"])
                - roc_auc_score(sample["label"], sample["p_real"])
            )
        )
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
