"""Paper-compatible pooled grayscale histogram fidelity metrics (lines 269-283)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FidelityMetrics:
    """Directional KL, Jensen-Shannon, and distribution-distance outputs."""

    kl_real_syn: float
    kl_syn_real: float
    js_divergence: float
    js_distance: float
    brier_dist: float
    rmse_dist: float

    def to_dict(self) -> dict[str, float]:
        """Return stable names for JSON reporting."""
        return asdict(self)


def pooled_histogram(paths: list[Path]) -> np.ndarray:
    """Pool raw pixels without resizing or equal image weighting.

    This intentionally preserves the paper's pixel-count weighting quirk: larger images
    contribute more observations. Changing it would prevent reproduction of lines 269-283.
    """
    if not paths:
        raise ValueError("at least one image is required")
    counts = np.zeros(256, dtype=np.float64)
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        counts += np.histogram(pixels, bins=256, range=(0, 256))[0]
    counts += 1e-12
    return cast(np.ndarray, counts / counts.sum())


def histogram_metrics(real: np.ndarray, synthetic: np.ndarray) -> FidelityMetrics:
    """Compare two normalized histograms with all separately named paper metrics."""
    first = np.asarray(real, dtype=np.float64)
    second = np.asarray(synthetic, dtype=np.float64)
    if first.shape != (256,) or second.shape != (256,):
        raise ValueError("fidelity histograms must each have 256 bins")
    if np.any(first <= 0) or np.any(second <= 0):
        raise ValueError("histograms must be positive; apply the prescribed epsilon first")
    first = first / first.sum()
    second = second / second.sum()
    midpoint = (first + second) / 2
    kl_real_syn = float(np.sum(first * np.log(first / second)))
    kl_syn_real = float(np.sum(second * np.log(second / first)))
    js_divergence = float(
        0.5 * np.sum(first * np.log(first / midpoint))
        + 0.5 * np.sum(second * np.log(second / midpoint))
    )
    brier = float(np.mean((first - second) ** 2))
    return FidelityMetrics(
        kl_real_syn,
        kl_syn_real,
        js_divergence,
        math.sqrt(max(js_divergence, 0)),
        brier,
        math.sqrt(brier),
    )


def evaluate_fidelity(real_paths: list[Path], synthetic_paths: list[Path]) -> FidelityMetrics:
    """Load pooled grayscale distributions and calculate fidelity metrics."""
    return histogram_metrics(pooled_histogram(real_paths), pooled_histogram(synthetic_paths))
