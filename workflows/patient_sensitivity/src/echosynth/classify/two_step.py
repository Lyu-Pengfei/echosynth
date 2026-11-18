"""Two-stage CA probability composition from paper lines 221-226."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class TwoStepResult:
    """Image-level stage outputs and their composed CA score."""

    p_abnormal: np.ndarray
    p_ca_given_abnormal: np.ndarray
    p_ca: np.ndarray


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return cast(np.ndarray, np.log(clipped / (1 - clipped)))


def compose_two_step(
    p_abnormal: np.ndarray,
    p_ca_given_abnormal: np.ndarray,
    mode: str = "product",
) -> TwoStepResult:
    """Compose per image; the confirmed project default is the probability product."""
    first = np.asarray(p_abnormal, dtype=float)
    second = np.asarray(p_ca_given_abnormal, dtype=float)
    if first.shape != second.shape:
        raise ValueError("two-step stage arrays must have identical shape")
    if np.any((first < 0) | (first > 1) | (second < 0) | (second > 1)):
        raise ValueError("two-step probabilities must be in [0, 1]")
    if mode == "product":
        composed = first * second
    elif mode == "stage2_only":
        composed = second.copy()
    elif mode == "logit_sum":
        logit = _logit(first) + _logit(second)
        composed = 1 / (1 + np.exp(-logit))
    else:
        raise ValueError(f"unknown two-step composition: {mode}")
    return TwoStepResult(first, second, composed)


class TwoStepClassifier:
    """Compose predictions from independently trained stage callables."""

    def __init__(
        self,
        stage1: Callable[[np.ndarray], np.ndarray],
        stage2: Callable[[np.ndarray], np.ndarray],
        compose: str = "product",
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.compose = compose

    def predict(self, images: np.ndarray) -> TwoStepResult:
        """Compose stage probabilities per image before encounter aggregation."""
        return compose_two_step(self.stage1(images), self.stage2(images), self.compose)
