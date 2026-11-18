"""Shared utilities for reviewer-requested PRCC analyses."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "config.yaml"
    with source.open() as stream:
        config = yaml.safe_load(stream)
    config["package_root"] = str(ROOT)
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["project_root"]) / path


def absolutize_image_paths(
    frame: pd.DataFrame,
    base: Path,
    columns: tuple[str, ...] = ("png_path", "frame_path", "image_path"),
) -> pd.DataFrame:
    """Resolve manifest image paths exactly as the source project would."""
    result = frame.copy()
    for column in columns:
        if column not in result:
            continue
        result[column] = result[column].map(
            lambda value: str(
                Path(str(value))
                if Path(str(value)).is_absolute()
                else (base / str(value)).resolve()
            )
        )
    return result


def output_path(*parts: str) -> Path:
    path = ROOT / "results" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def normalize_vendor(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "UNKNOWN"}:
        return "UNKNOWN"
    aliases = {
        "GE MEDICAL SYSTEMS": "GE",
        "GE HEALTHCARE": "GE",
        "GENERAL ELECTRIC": "GE",
        "PHILIPS MEDICAL SYSTEMS": "PHILIPS",
        "PHILIPS HEALTHCARE": "PHILIPS",
        "SIEMENS HEALTHCARE": "SIEMENS",
        "SIEMENS MEDICAL SOLUTIONS": "SIEMENS",
    }
    for prefix, replacement in aliases.items():
        if prefix in text:
            return replacement
    return text.split("^")[0][:40]


def patient_metadata(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create one acquisition-summary row per held-out patient."""
    required = {"patient_id", "group", "split", "acquisition_year", "manufacturer"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing metadata columns: {sorted(missing)}")
    frame = manifest.loc[manifest["split"].astype(str).eq("test")].copy()
    frame["acquisition_year"] = pd.to_numeric(frame["acquisition_year"], errors="coerce")
    frame["vendor"] = frame["manufacturer"].map(normalize_vendor)

    def mode(series: pd.Series) -> str:
        values = series.dropna().astype(str)
        if values.empty:
            return "UNKNOWN"
        counts = values.value_counts()
        return str(sorted(counts[counts.eq(counts.max())].index)[0])

    grouped = frame.groupby("patient_id", observed=True)
    output = grouped.agg(
        group=("group", "first"),
        acquisition_year=("acquisition_year", "median"),
        vendor=("vendor", mode),
        n_images=("image_id", "nunique"),
        n_encounters=("encounter_id", "nunique"),
    ).reset_index()
    output["label"] = output["group"].astype(str).eq("CA").astype(int)
    return output


def bootstrap_metric(
    frame: pd.DataFrame,
    metric,
    columns: list[str],
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    point = float(metric(*(frame[column].to_numpy() for column in columns)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(len(frame))
    for _ in range(replicates):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        sample = frame.iloc[chosen]
        if "label" in sample and sample["label"].nunique() < 2:
            continue
        try:
            values.append(float(metric(*(sample[column].to_numpy() for column in columns))))
        except ValueError:
            continue
    if not values:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def auc_summary(frame: pd.DataFrame, probability: str, replicates: int, seed: int) -> dict[str, float]:
    auc, low, high = bootstrap_metric(
        frame, roc_auc_score, ["label", probability], replicates, seed
    )
    return {
        "n": int(len(frame)),
        "n_positive": int(frame["label"].sum()),
        "n_negative": int((1 - frame["label"]).sum()),
        "auroc": auc,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "auprc": float(average_precision_score(frame["label"], frame[probability])),
    }


def paired_auc_delta(
    frame: pd.DataFrame,
    reference: str,
    comparison: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    def delta(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float(roc_auc_score(y, second) - roc_auc_score(y, first))

    point, low, high = bootstrap_metric(
        frame, delta, ["label", reference, comparison], replicates, seed
    )
    return {"delta_auroc": point, "ci_low": low, "ci_high": high}


def resolve_image_path(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_file():
        return path
    candidate = manifest_path.parent / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(path)


def image_tensor(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return torch.from_numpy(array / 127.5 - 1.0).unsqueeze(0)


def stable_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy().reset_index(drop=True)
    keys = frame.apply(
        lambda row: hashlib.sha256(
            f"{seed}|{row.get('image_id', row.name)}".encode()
        ).hexdigest(),
        axis=1,
    )
    return frame.loc[keys.sort_values().index[:count]].reset_index(drop=True)


def covariance_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    mean_first = first.mean(axis=0)
    mean_second = second.mean(axis=0)
    cov_first = np.atleast_2d(np.cov(first, rowvar=False))
    cov_second = np.atleast_2d(np.cov(second, rowvar=False))
    root = covariance_sqrt(cov_first)
    middle = covariance_sqrt(root @ cov_second @ root)
    value = (
        np.square(mean_first - mean_second).sum()
        + np.trace(cov_first)
        + np.trace(cov_second)
        - 2.0 * np.trace(middle)
    )
    return float(max(value, 0.0))


def rbf_mmd(first: np.ndarray, second: np.ndarray, max_rows: int = 1000) -> float:
    first = first[:max_rows]
    second = second[:max_rows]
    joined = np.concatenate([first, second], axis=0)
    squared = np.square(joined[:, None, :] - joined[None, :, :]).sum(axis=2)
    nonzero = squared[squared > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    kernel = np.exp(-squared / max(2.0 * bandwidth, 1e-12))
    n = len(first)
    k_xx = kernel[:n, :n]
    k_yy = kernel[n:, n:]
    k_xy = kernel[:n, n:]
    first_term = (k_xx.sum() - np.trace(k_xx)) / max(n * (n - 1), 1)
    m = len(second)
    second_term = (k_yy.sum() - np.trace(k_yy)) / max(m * (m - 1), 1)
    return float(first_term + second_term - 2.0 * k_xy.mean())


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
