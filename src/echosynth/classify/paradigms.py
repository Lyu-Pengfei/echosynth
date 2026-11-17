"""Leakage-safe TRTR, TSTR, TRSTR, and matched-bootstrap experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from echosynth.classify.engine import train_classifier
from echosynth.classify.models import CompactCNN
from echosynth.classify.transforms import build_transform
from echosynth.classify.two_step import compose_two_step
from echosynth.data.datasets import EchoClassificationDataset
from echosynth.data.manifest import assert_manifest_invariants, read_manifest
from echosynth.eval.aggregate import to_patient_level
from echosynth.eval.delong import delong_ci, delong_paired_delta_ci
from echosynth.eval.guard import TestSetGuard
from echosynth.eval.metrics import auprc, auroc
from echosynth.utils.checkpoint import load_checkpoint
from echosynth.utils.config import Config, git_sha, save_resolved
from echosynth.utils.io import atomic_write_json, atomic_write_text
from echosynth.utils.seed import set_seed

PARADIGMS = ("TRTR", "TSTR", "TRSTR", "BOOTSTRAP")


@dataclass(frozen=True)
class SelectionPoint:
    """One synthetic-count validation result."""

    count: int
    val_auroc: float


@dataclass(frozen=True)
class ExperimentResult:
    """Paths and metrics produced by one fully resolved experiment."""

    run_id: str
    run_dir: Path
    selected_synthetic_count: int
    metrics: dict[str, Any]
    selection_curve: tuple[SelectionPoint, ...]


@dataclass
class _TaskModels:
    stage1: CompactCNN
    stage2: CompactCNN | None


def augment_training_data(
    real_train: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    paradigm: str,
    count: int,
    *,
    minority_label: str = "CA",
    seed: int = 2024,
) -> pd.DataFrame:
    """Construct one paradigm's training rows without changing the majority class."""
    if paradigm not in PARADIGMS:
        raise ValueError(f"unknown paradigm: {paradigm}")
    if count < 0:
        raise ValueError("synthetic count must be non-negative")
    real = real_train.copy()
    synthetic = synthetic_train.copy()
    if "class_label" in synthetic and "group" not in synthetic:
        synthetic["group"] = synthetic["class_label"]
    if "split" in synthetic and not synthetic["split"].astype(str).eq("train").all():
        raise ValueError("synthetic rows must all have split='train'")
    candidates = synthetic.loc[
        synthetic.get("group", pd.Series(index=synthetic.index)).astype(str) == minority_label
    ]
    if paradigm in {"TRSTR", "TSTR"} and len(candidates) < count:
        raise ValueError(
            f"requested {count} synthetic {minority_label} rows but only {len(candidates)} exist"
        )
    chosen = (
        candidates.sample(n=count, random_state=seed, replace=False)
        if count
        else candidates.iloc[:0]
    )
    if paradigm == "TRTR":
        return real.reset_index(drop=True)
    if paradigm == "TSTR":
        other = synthetic.loc[synthetic.get("group", "").astype(str) != minority_label]
        return pd.concat([chosen, other], ignore_index=True)
    if paradigm == "TRSTR":
        return pd.concat([real, chosen], ignore_index=True)
    minority = real.loc[real["group"].astype(str) == minority_label]
    if count and minority.empty:
        raise ValueError(f"cannot bootstrap absent minority class {minority_label}")
    sampled = (
        minority.sample(n=count, replace=True, random_state=seed) if count else minority.iloc[:0]
    )
    return pd.concat([real, sampled], ignore_index=True)


def select_synthetic_count(
    counts: Sequence[int], validation_scorer: Callable[[int], float]
) -> tuple[int, tuple[SelectionPoint, ...]]:
    """Select the count from validation scores only; ties choose the smaller count."""
    if not counts:
        raise ValueError("synthetic count grid cannot be empty")
    points = tuple(
        SelectionPoint(int(count), float(validation_scorer(int(count)))) for count in counts
    )
    selected = max(points, key=lambda point: (point.val_auroc, -point.count)).count
    return selected, points


def _normalize_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "class_label" in result:
        result["group"] = result["class_label"].astype(str)
    if "encounter_id" not in result:
        result["encounter_id"] = "synthetic-" + result["image_id"].astype(str)
    if "patient_id" not in result:
        result["patient_id"] = result["encounter_id"]
    result["split"] = "train"
    return result


def _loader(
    frame: pd.DataFrame,
    training: bool,
    image_size: int,
    batch_size: int,
) -> DataLoader[Any]:
    dataset = EchoClassificationDataset(
        frame,
        "_target",
        {"0": 0, "1": 1},
        build_transform(training, image_size),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=False)


def _fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    out_dir: Path,
    settings: Mapping[str, Any],
    seed: int,
) -> CompactCNN:
    model = CompactCNN(
        channels=tuple(settings.get("channels", (32, 64, 128, 256))),
        dropout=float(settings.get("dropout", 0.3)),
    )
    image_size = int(settings.get("image_size", 112))
    batch_size = int(settings.get("batch_size", 32))
    result = train_classifier(
        model,
        _loader(train, True, image_size, batch_size),
        _loader(validation, False, image_size, batch_size),
        out_dir,
        epochs=int(settings.get("epochs", 100)),
        patience=int(settings.get("patience", 15)),
        lr=float(settings.get("lr", 1e-3)),
        seed=seed,
    )
    state, _ = load_checkpoint(result.checkpoint)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _add_target(frame: pd.DataFrame, task: str, stage: int) -> pd.DataFrame:
    output = frame.copy()
    groups = output["group"].astype(str)
    if task == "two_step" and stage == 1:
        output["_target"] = groups.isin(["CA", "LVH"]).astype(int).astype(str)
    elif task == "two_step" and stage == 2:
        output = output.loc[groups.isin(["CA", "LVH"])].copy()
        output["_target"] = output["group"].astype(str).eq("CA").astype(int).astype(str)
    else:
        output["_target"] = groups.eq("CA").astype(int).astype(str)
    if output["_target"].nunique() != 2:
        raise ValueError(
            f"training/evaluation stage lacks both classes for task={task}, stage={stage}"
        )
    return output


def _fit_task(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    out_dir: Path,
    settings: Mapping[str, Any],
    task: str,
    seed: int,
) -> _TaskModels:
    stage1 = _fit_binary(
        _add_target(train, task, 1),
        _add_target(validation, task, 1),
        out_dir / "stage1",
        settings,
        seed,
    )
    stage2 = None
    if task == "two_step":
        stage2 = _fit_binary(
            _add_target(train, task, 2),
            _add_target(validation, task, 2),
            out_dir / "stage2",
            settings,
            seed + 1,
        )
    return _TaskModels(stage1, stage2)


def _image_probabilities(
    model: CompactCNN, frame: pd.DataFrame, settings: Mapping[str, Any]
) -> np.ndarray:
    eval_frame = frame.copy()
    eval_frame["_target"] = "0"
    loader = _loader(
        eval_frame,
        False,
        int(settings.get("image_size", 112)),
        int(settings.get("batch_size", 32)),
    )
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in loader:
            probabilities.append(torch.softmax(model(images), dim=1)[:, 1].numpy())
    return np.concatenate(probabilities)


def _predict_task(
    models: _TaskModels,
    frame: pd.DataFrame,
    settings: Mapping[str, Any],
    task: str,
    compose: str,
) -> pd.DataFrame:
    first = _image_probabilities(models.stage1, frame, settings)
    probabilities = first
    if task == "two_step":
        if models.stage2 is None:
            raise RuntimeError("two-step task has no stage-2 model")
        second = _image_probabilities(models.stage2, frame, settings)
        probabilities = compose_two_step(first, second, compose).p_ca
    output = frame[["encounter_id", "patient_id", "group", "split"]].copy()
    output["label"] = frame["group"].astype(str).eq("CA").astype(int)
    output["p"] = probabilities
    return to_patient_level(output)


def _grid(settings: Mapping[str, Any]) -> list[int]:
    start, stop, step = (int(settings[key]) for key in ("start", "stop", "step"))
    if step <= 0 or stop < start:
        raise ValueError("invalid synthetic grid")
    return list(range(start, stop + 1, step))


def run_experiment(config: Config | Mapping[str, Any]) -> ExperimentResult:
    """Select every model/count on validation, then read and score test once."""
    values = config.to_dict() if isinstance(config, Config) else dict(config)
    seed = int(values.get("seed", 2024))
    set_seed(seed)
    name = str(values.get("name", "experiment"))
    run_id = str(values.get("run_id", f"{name}-seed-{seed}"))
    results_root = Path(values.get("results_dir", "results/runs"))
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(values["data"]["manifest"])
    real = read_manifest(manifest_path)
    assert_manifest_invariants(real)
    train = real.loc[real["split"].astype(str) == "train"].copy()
    validation = real.loc[real["split"].astype(str) == "val"].copy()
    synthetic = _normalize_synthetic(pd.read_csv(values["synthetic_manifest"]))
    settings = dict(values.get("classifier", {}))
    settings.setdefault("image_size", values.get("data", {}).get("image_size", 112))
    task_values = dict(values.get("task", {}))
    task = str(task_values.get("type", "binary_ca"))
    compose = str(task_values.get("compose", "product"))
    paradigms = [str(item) for item in values.get("paradigms", ["TRTR", "TRSTR", "BOOTSTRAP"])]
    if any(item not in PARADIGMS for item in paradigms):
        raise ValueError(f"invalid paradigms: {paradigms}")
    models: dict[str, _TaskModels] = {}
    baseline_train = augment_training_data(train, synthetic, "TRTR", 0, seed=seed)
    models["TRTR"] = _fit_task(
        baseline_train, validation, run_dir / "ckpt/TRTR", settings, task, seed
    )
    grid_models: dict[int, _TaskModels] = {}

    def validation_score(count: int) -> float:
        arm_train = augment_training_data(train, synthetic, "TRSTR", count, seed=seed + count)
        candidate = _fit_task(
            arm_train,
            validation,
            run_dir / f"ckpt/TRSTR-{count}",
            settings,
            task,
            seed + count,
        )
        grid_models[count] = candidate
        predictions = _predict_task(candidate, validation, settings, task, compose)
        return auroc(predictions["label"], predictions["p"])

    selected_count, curve = select_synthetic_count(
        _grid(values["synthetic_grid"]), validation_score
    )
    models["TRSTR"] = grid_models[selected_count]
    if "BOOTSTRAP" in paradigms:
        bootstrap = augment_training_data(
            train, synthetic, "BOOTSTRAP", selected_count, seed=seed + 991
        )
        models["BOOTSTRAP"] = _fit_task(
            bootstrap,
            validation,
            run_dir / "ckpt/BOOTSTRAP",
            settings,
            task,
            seed + 991,
        )
    if "TSTR" in paradigms:
        synthetic_only = augment_training_data(
            train, synthetic, "TSTR", selected_count, seed=seed + 313
        )
        models["TSTR"] = _fit_task(
            synthetic_only,
            validation,
            run_dir / "ckpt/TSTR",
            settings,
            task,
            seed + 313,
        )
    selected_models = {key: model for key, model in models.items() if key in paradigms}
    prediction_frames: dict[str, pd.DataFrame] = {}
    with TestSetGuard(run_dir / "test_access.jsonl"):
        test = real.loc[real["split"].astype(str) == "test"].copy()
        for paradigm, model in selected_models.items():
            prediction_frames[paradigm] = _predict_task(model, test, settings, task, compose)
    metrics: dict[str, Any] = {}
    for paradigm, predictions in prediction_frames.items():
        auc, ci_low, ci_high = delong_ci(predictions["label"], predictions["p"])
        metrics[paradigm] = {
            "auroc": auc,
            "auroc_ci": [ci_low, ci_high],
            "auprc": auprc(predictions["label"], predictions["p"]),
            "n_patients": len(predictions),
        }
        predictions.to_csv(run_dir / f"predictions_{paradigm}.csv", index=False)
    if "TRTR" in prediction_frames:
        baseline = prediction_frames["TRTR"]
        for paradigm, predictions in prediction_frames.items():
            if paradigm == "TRTR":
                continue
            merged = baseline[["patient_id", "label", "p"]].merge(
                predictions[["patient_id", "p"]],
                on="patient_id",
                suffixes=("_trtr", "_arm"),
            )
            delta = delong_paired_delta_ci(merged["label"], merged["p_trtr"], merged["p_arm"])
            metrics[paradigm]["delta_vs_trtr"] = {
                "delta": delta.delta,
                "ci": [delta.ci_low, delta.ci_high],
                "z": delta.z,
                "p_value": delta.p_value,
            }
    metrics["selected_synthetic_count"] = selected_count
    atomic_write_json(run_dir / "metrics.json", metrics)
    pd.DataFrame([point.__dict__ for point in curve]).to_csv(
        run_dir / "selection_curve.csv", index=False
    )
    if isinstance(config, Config):
        save_resolved(config, run_dir)
    else:
        import yaml

        atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(values, sort_keys=True))
        atomic_write_text(run_dir / "git_sha.txt", git_sha() + "\n")
    atomic_write_text(
        run_dir / "log.txt",
        f"run_id={run_id}\nseed={seed}\nselected_synthetic_count={selected_count}\n",
    )
    return ExperimentResult(run_id, run_dir, selected_count, metrics, curve)
