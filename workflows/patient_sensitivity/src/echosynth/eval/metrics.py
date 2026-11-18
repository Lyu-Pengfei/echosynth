"""Tested sklearn metric wrappers with the paper's AUPRC point-estimate policy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def auroc(y: Sequence[int], p: Sequence[float]) -> float:
    """Return binary area under the ROC curve."""
    return float(roc_auc_score(y, p))


def auprc(y: Sequence[int], p: Sequence[float]) -> float:
    """Return binary average precision as the AUPRC point estimate."""
    return float(average_precision_score(y, p))


def clinical_binary_metrics(
    y: Sequence[int], p: Sequence[float], threshold: float = 0.5, bins: int = 10
) -> dict[str, float | int]:
    """Return prevalence, Brier score, ECE, and fixed-threshold operating metrics."""
    labels = np.asarray(y, dtype=int)
    probabilities = np.asarray(p, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        include_right = index == bins - 1
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if include_right
            else probabilities < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return {
        "n": int(len(labels)),
        "prevalence": float(labels.mean()),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10bin": float(ece),
        "threshold": float(threshold),
        "sensitivity": ratio(int(tp), int(tp + fn)),
        "specificity": ratio(int(tn), int(tn + fp)),
        "ppv": ratio(int(tp), int(tp + fp)),
        "npv": ratio(int(tn), int(tn + fn)),
    }


def auprc_ci(*_args: object, **_kwargs: object) -> tuple[float, float, float]:
    """Refuse an AUPRC CI because paper line 239 reports a point estimate only."""
    raise NotImplementedError("AUPRC confidence intervals are not defined in v1 (paper line 239)")
