"""Sun and Xu (2014) fast DeLong AUROC variance and paired comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class DeltaResult:
    """Paired p2-minus-p1 AUROC delta with interval, z statistic, and p-value."""

    delta: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    output = np.empty(len(values), dtype=float)
    output[order] = ranks + 1
    return output


def _covariance(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] < 2:
        raise ValueError("DeLong variance requires at least two samples in each class")
    covariance = np.cov(matrix, bias=False)
    return np.atleast_2d(covariance)


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, total = predictions.shape
    negative_count = total - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong requires at least two positive and two negative observations")
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    positive_ranks = np.empty_like(positive, dtype=float)
    negative_ranks = np.empty_like(negative, dtype=float)
    total_ranks = np.empty_like(predictions, dtype=float)
    for classifier in range(classifiers):
        positive_ranks[classifier] = _midranks(positive[classifier])
        negative_ranks[classifier] = _midranks(negative[classifier])
        total_ranks[classifier] = _midranks(predictions[classifier])
    aucs = total_ranks[:, :positive_count].sum(axis=1) / (positive_count * negative_count) - (
        positive_count + 1
    ) / (2 * negative_count)
    v01 = (total_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1 - (total_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = _covariance(v01) / positive_count + _covariance(v10) / negative_count
    return aucs, covariance


def _prepare(y: Sequence[int], predictions: Sequence[Sequence[float]]) -> tuple[np.ndarray, int]:
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "DeLong labels must be one-dimensional binary values containing both classes"
        )
    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("prediction matrix must have shape (classifiers, observations)")
    if not np.isfinite(matrix).all():
        raise ValueError("predictions must be finite")
    order = np.argsort(-labels, kind="stable")
    return matrix[:, order], int(labels.sum())


def delong_auc_variance(y: Sequence[int], p: Sequence[float]) -> tuple[float, float]:
    """Return AUROC and Sun-Xu fast DeLong variance."""
    matrix, positives = _prepare(y, [p])
    aucs, covariance = _fast_delong(matrix, positives)
    return float(aucs[0]), float(covariance[0, 0])


def delong_ci(
    y: Sequence[int], p: Sequence[float], alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return AUROC and a clipped normal-approximation DeLong interval."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    auc, variance = delong_auc_variance(y, p)
    quantile = float(norm.ppf(1 - alpha / 2))
    margin = quantile * math.sqrt(max(variance, 0))
    return auc, max(0.0, auc - margin), min(1.0, auc + margin)


def normal_delta_ci(
    delta: float, standard_error: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Construct an unbounded paired-delta interval from its paired standard error."""
    if standard_error < 0 or not 0 < alpha < 1:
        raise ValueError("standard_error must be non-negative and alpha in (0, 1)")
    margin = float(norm.ppf(1 - alpha / 2)) * standard_error
    return delta - margin, delta + margin


def delong_paired_delta_ci(
    y: Sequence[int],
    p1: Sequence[float],
    p2: Sequence[float],
    alpha: float = 0.05,
) -> DeltaResult:
    """Return the paired p2-minus-p1 AUROC delta and DeLong interval/test."""
    matrix, positives = _prepare(y, [p1, p2])
    aucs, covariance = _fast_delong(matrix, positives)
    contrast = np.array([-1.0, 1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[1] - aucs[0])
    standard_error = math.sqrt(max(variance, 0))
    ci_low, ci_high = normal_delta_ci(delta, standard_error, alpha)
    if standard_error == 0:
        z = 0.0 if delta == 0 else math.copysign(math.inf, delta)
        p_value = 1.0 if delta == 0 else 0.0
    else:
        z = delta / standard_error
        p_value = float(2 * norm.sf(abs(z)))
    return DeltaResult(delta, ci_low, ci_high, z, p_value)
