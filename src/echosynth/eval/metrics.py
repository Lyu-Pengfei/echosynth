"""Tested sklearn metric wrappers with the paper's AUPRC point-estimate policy."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y: Sequence[int], p: Sequence[float]) -> float:
    """Return binary area under the ROC curve."""
    return float(roc_auc_score(y, p))


def auprc(y: Sequence[int], p: Sequence[float]) -> float:
    """Return binary average precision as the AUPRC point estimate."""
    return float(average_precision_score(y, p))


def auprc_ci(*_args: object, **_kwargs: object) -> tuple[float, float, float]:
    """Refuse an AUPRC CI because paper line 239 reports a point estimate only."""
    raise NotImplementedError("AUPRC confidence intervals are not defined in v1 (paper line 239)")
