"""Leakage-safe TRTR, TSTR, TRSTR, and matched-bootstrap experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from echosynth.classify.engine import train_classifier
from echosynth.classify.models import CompactCNN
from echosynth.classify.transforms import build_transform
from echosynth.classify.two_step import compose_two_step
from echosynth.data.datasets import EchoClassificationDataset
from echosynth.data.manifest import assert_manifest_invariants, read_manifest
from echosynth.eval.aggregate import to_encounter_level
from echosynth.eval.delong import delong_ci, delong_paired_delta_ci
from echosynth.eval.guard import TestSetGuard
from echosynth.eval.metrics import auprc, auroc, clinical_binary_metrics
from echosynth.utils.checkpoint import load_checkpoint
from echosynth.utils.config import Config, git_sha, save_resolved
from echosynth.utils.io import atomic_write_json, atomic_write_text
from echosynth.utils.seed import set_seed

PARADIGMS = (
    "TRTR",
    "TSTR",
    "TRSTR",
    "BOOTSTRAP",
    "CLASS_WEIGHTED",
    "BALANCED_SAMPLING",
)


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
    if paradigm in {"TRTR", "CLASS_WEIGHTED", "BALANCED_SAMPLING"}:
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
    *,
    balanced: bool = False,
    num_workers: int = 0,
) -> DataLoader[Any]:
    dataset = EchoClassificationDataset(
        frame,
        "_target",
        {"0": 0, "1": 1},
        build_transform(training, image_size),
    )
    sampler = None
    if training and balanced:
        labels = frame["_target"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError("balanced sampling requires both binary classes")
        weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    out_dir: Path,
    settings: Mapping[str, Any],
    seed: int,
    training_mode: str = "standard",
) -> CompactCNN:
    model = CompactCNN(
        channels=tuple(settings.get("channels", (32, 64, 128, 256))),
        dropout=float(settings.get("dropout", 0.3)),
    )
    image_size = int(settings.get("image_size", 112))
    batch_size = int(settings.get("batch_size", 32))
    num_workers = int(settings.get("num_workers", 0))
    class_weights = None
    if training_mode == "class_weighted":
        labels = train["_target"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError("class-weighted loss requires both binary classes")
        class_weights = torch.as_tensor(
            len(labels) / (2.0 * counts), dtype=torch.float32
        )
    result = train_classifier(
        model,
        _loader(
            train,
            True,
            image_size,
            batch_size,
            balanced=training_mode == "balanced_sampling",
            num_workers=num_workers,
        ),
        _loader(validation, False, image_size, batch_size, num_workers=num_workers),
        out_dir,
        epochs=int(settings.get("epochs", 100)),
        patience=int(settings.get("patience", 15)),
        lr=float(settings.get("lr", 1e-3)),
        seed=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        class_weights=class_weights,
    )
    state, _ = load_checkpoint(result.checkpoint)
    model.load_state_dict(state["model"])
    model.eval()
    model.to("cpu")
    return model


def _add_target(
    frame: pd.DataFrame,
    task: str,
    stage: int,
    positive_label: str,
    intermediate_label: str,
) -> pd.DataFrame:
    output = frame.copy()
    groups = output["group"].astype(str)
    if task == "two_step" and stage == 1:
        output["_target"] = (
            groups.isin([positive_label, intermediate_label]).astype(int).astype(str)
        )
    elif task == "two_step" and stage == 2:
        output = output.loc[groups.isin([positive_label, intermediate_label])].copy()
        output["_target"] = (
            output["group"].astype(str).eq(positive_label).astype(int).astype(str)
        )
    else:
        output["_target"] = groups.eq(positive_label).astype(int).astype(str)
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
    positive_label: str,
    intermediate_label: str,
    training_mode: str = "standard",
) -> _TaskModels:
    stage1 = _fit_binary(
        _add_target(train, task, 1, positive_label, intermediate_label),
        _add_target(validation, task, 1, positive_label, intermediate_label),
        out_dir / "stage1",
        settings,
        seed,
        training_mode,
    )
    stage2 = None
    if task == "two_step":
        stage2 = _fit_binary(
            _add_target(train, task, 2, positive_label, intermediate_label),
            _add_target(validation, task, 2, positive_label, intermediate_label),
            out_dir / "stage2",
            settings,
            seed + 1,
            training_mode,
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
        num_workers=int(settings.get("num_workers", 0)),
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
    positive_label: str,
) -> pd.DataFrame:
    first = _image_probabilities(models.stage1, frame, settings)
    probabilities = first
    if task == "two_step":
        if models.stage2 is None:
            raise RuntimeError("two-step task has no stage-2 model")
        second = _image_probabilities(models.stage2, frame, settings)
        probabilities = compose_two_step(first, second, compose).p_ca
    output = frame[["encounter_id", "patient_id", "group", "split"]].copy()
    output["label"] = frame["group"].astype(str).eq(positive_label).astype(int)
    output["p"] = probabilities
    return to_encounter_level(output)


def _predict_stage(
    model: CompactCNN,
    frame: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    positive_groups: set[str],
    include_groups: set[str] | None = None,
) -> pd.DataFrame:
    selected = frame.copy()
    if include_groups is not None:
        selected = selected.loc[selected["group"].astype(str).isin(include_groups)].copy()
    probabilities = _image_probabilities(model, selected, settings)
    output = selected[["encounter_id", "patient_id", "group", "split"]].copy()
    output["label"] = selected["group"].astype(str).isin(positive_groups).astype(int)
    output["p"] = probabilities
    return to_encounter_level(output)


def _metric_block(predictions: pd.DataFrame) -> dict[str, Any]:
    try:
        auc, ci_low, ci_high = delong_ci(predictions["label"], predictions["p"])
    except ValueError:
        # Tiny smoke cohorts may have only one observation in a class. The point
        # estimate is still useful for pipeline verification; a CI is not estimable.
        auc = auroc(predictions["label"], predictions["p"])
        ci_low, ci_high = float("nan"), float("nan")
    return {
        "auroc": auc,
        "auroc_ci": [ci_low, ci_high],
        "auprc": auprc(predictions["label"], predictions["p"]),
        "n_patients": int(predictions["patient_id"].nunique()),
        "n_encounters": len(predictions),
        **clinical_binary_metrics(predictions["label"], predictions["p"]),
    }


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
    positive_label = str(task_values.get("positive_label", "CA"))
    intermediate_label = str(task_values.get("intermediate_label", "LVH"))
    paradigms = [str(item) for item in values.get("paradigms", ["TRTR", "TRSTR", "BOOTSTRAP"])]
    if any(item not in PARADIGMS for item in paradigms):
        raise ValueError(f"invalid paradigms: {paradigms}")
    models: dict[str, _TaskModels] = {}
    baseline_train = augment_training_data(
        train, synthetic, "TRTR", 0, minority_label=positive_label, seed=seed
    )
    models["TRTR"] = _fit_task(
        baseline_train,
        validation,
        run_dir / "ckpt/TRTR",
        settings,
        task,
        seed,
        positive_label,
        intermediate_label,
    )
    grid_models: dict[int, _TaskModels] = {}

    def validation_score(count: int) -> float:
        arm_train = augment_training_data(
            train,
            synthetic,
            "TRSTR",
            count,
            minority_label=positive_label,
            seed=seed + count,
        )
        candidate = _fit_task(
            arm_train,
            validation,
            run_dir / f"ckpt/TRSTR-{count}",
            settings,
            task,
            seed + count,
            positive_label,
            intermediate_label,
        )
        grid_models[count] = candidate
        predictions = _predict_task(
            candidate, validation, settings, task, compose, positive_label
        )
        return auroc(predictions["label"], predictions["p"])

    needs_synthetic_selection = any(
        item in paradigms for item in ("TRSTR", "BOOTSTRAP", "TSTR")
    )
    if needs_synthetic_selection:
        selected_count, curve = select_synthetic_count(
            _grid(values["synthetic_grid"]), validation_score
        )
        models["TRSTR"] = grid_models[selected_count]
    else:
        selected_count, curve = 0, tuple()
    if "CLASS_WEIGHTED" in paradigms:
        models["CLASS_WEIGHTED"] = _fit_task(
            baseline_train,
            validation,
            run_dir / "ckpt/CLASS_WEIGHTED",
            settings,
            task,
            seed + 1777,
            positive_label,
            intermediate_label,
            "class_weighted",
        )
    if "BALANCED_SAMPLING" in paradigms:
        models["BALANCED_SAMPLING"] = _fit_task(
            baseline_train,
            validation,
            run_dir / "ckpt/BALANCED_SAMPLING",
            settings,
            task,
            seed + 2777,
            positive_label,
            intermediate_label,
            "balanced_sampling",
        )
    if "BOOTSTRAP" in paradigms:
        bootstrap = augment_training_data(
            train,
            synthetic,
            "BOOTSTRAP",
            selected_count,
            minority_label=positive_label,
            seed=seed + 991,
        )
        models["BOOTSTRAP"] = _fit_task(
            bootstrap,
            validation,
            run_dir / "ckpt/BOOTSTRAP",
            settings,
            task,
            seed + 991,
            positive_label,
            intermediate_label,
        )
    if "TSTR" in paradigms:
        synthetic_only = augment_training_data(
            train,
            synthetic,
            "TSTR",
            selected_count,
            minority_label=positive_label,
            seed=seed + 313,
        )
        models["TSTR"] = _fit_task(
            synthetic_only,
            validation,
            run_dir / "ckpt/TSTR",
            settings,
            task,
            seed + 313,
            positive_label,
            intermediate_label,
        )
    selected_models = {key: model for key, model in models.items() if key in paradigms}
    for paradigm, model in selected_models.items():
        validation_predictions = _predict_task(
            model, validation, settings, task, compose, positive_label
        )
        validation_predictions.to_csv(
            run_dir / f"validation_predictions_{paradigm}.csv", index=False
        )
        if task == "two_step" and model.stage2 is not None:
            validation_stage2 = _predict_stage(
                model.stage2,
                validation,
                settings,
                positive_groups={positive_label},
                include_groups={positive_label, intermediate_label},
            )
            validation_stage2.to_csv(
                run_dir / f"validation_predictions_stage2_{paradigm}.csv", index=False
            )
    prediction_frames: dict[str, pd.DataFrame] = {}
    stage1_frames: dict[str, pd.DataFrame] = {}
    stage2_frames: dict[str, pd.DataFrame] = {}
    with TestSetGuard(run_dir / "test_access.jsonl"):
        test = real.loc[real["split"].astype(str) == "test"].copy()
        for paradigm, model in selected_models.items():
            prediction_frames[paradigm] = _predict_task(
                model, test, settings, task, compose, positive_label
            )
            if task == "two_step":
                stage1_frames[paradigm] = _predict_stage(
                    model.stage1,
                    test,
                    settings,
                    positive_groups={positive_label, intermediate_label},
                )
                if model.stage2 is None:
                    raise RuntimeError("two-step task has no stage-2 model")
                stage2_frames[paradigm] = _predict_stage(
                    model.stage2,
                    test,
                    settings,
                    positive_groups={positive_label},
                    include_groups={positive_label, intermediate_label},
                )
    metrics: dict[str, Any] = {}
    for paradigm, predictions in prediction_frames.items():
        metrics[paradigm] = _metric_block(predictions)
        predictions.to_csv(run_dir / f"predictions_{paradigm}.csv", index=False)
        if paradigm in stage1_frames:
            metrics[paradigm]["stage1_abnormal_vs_general"] = _metric_block(
                stage1_frames[paradigm]
            )
            metrics[paradigm]["stage2_ca_vs_lvh"] = _metric_block(stage2_frames[paradigm])
            stage1_frames[paradigm].to_csv(
                run_dir / f"predictions_stage1_{paradigm}.csv", index=False
            )
            stage2_frames[paradigm].to_csv(
                run_dir / f"predictions_stage2_{paradigm}.csv", index=False
            )
    if "TRTR" in prediction_frames:
        baseline = prediction_frames["TRTR"]
        for paradigm, predictions in prediction_frames.items():
            if paradigm == "TRTR":
                continue
            merged = baseline[["encounter_id", "label", "p"]].merge(
                predictions[["encounter_id", "p"]], on="encounter_id", suffixes=("_trtr", "_arm")
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
