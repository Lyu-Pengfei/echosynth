"""Legacy target-range checks, not a demonstration of independent reproduction.

The withdrawn EchoNet-LVH paired interval is deliberately not a target. These
checks do not validate prediction provenance, CI estimation, or model equivalence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

TARGETS = {
    "duke_ca_trtr": (0.883, 0.831, 0.927),
    "duke_ca_bootstrap": (0.899, 0.856, 0.934),
    "duke_ca_trstr": (0.924, 0.883, 0.955),
    "duke_ca_delta": (0.041, 0.013, 0.069),
    "echonet_lvh_trtr": (0.832, 0.784, 0.874),
    "echonet_lvh_trstr": (0.864, 0.824, 0.901),
    "duke_tstr": (0.889, 0.842, 0.936),
    "discriminator_test": (0.612, 0.537, 0.692),
}


@pytest.mark.reproduction
def test_reported_targets_from_mounted_artifact() -> None:
    artifact_value = os.environ.get("ECHOSYNTH_REPRODUCTION_METRICS")
    if not artifact_value:
        pytest.skip("set ECHOSYNTH_REPRODUCTION_METRICS to a real-data metrics JSON artifact")
    metrics = json.loads(Path(artifact_value).read_text(encoding="utf-8"))
    for key, (point, low, high) in TARGETS.items():
        assert key in metrics, f"mounted artifact lacks {key}"
        observed = float(metrics[key])
        assert low <= observed <= high, f"{key}={observed} outside reported CI around {point}"
    fidelity = metrics["fidelity"]
    for key, target, tolerance in (
        ("kl_real_syn", 0.019, 0.003),
        ("kl_syn_real", 0.017, 0.003),
        ("js_divergence", 0.0043, 0.001),
        ("js_distance", 0.066, 0.01),
        ("brier_dist", 8.8e-6, 2e-6),
        ("rmse_dist", 0.0030, 5e-4),
    ):
        assert float(fidelity[key]) == pytest.approx(target, abs=tolerance)
