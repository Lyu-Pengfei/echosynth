"""Patient-level calibration diagnostics for the fixed five-seed v6 ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

from common import PARADIGMS, ensemble_predictions, patient_level, stratified_bootstrap


def calibration_parameters(frame: pd.DataFrame) -> tuple[float, float]:
    labels = frame["label"].to_numpy(dtype=float)
    probabilities = np.clip(frame["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logits
        return float(np.logaddexp(0.0, linear).sum() - np.dot(labels, linear))

    result = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.x[1])


def ece(frame: pd.DataFrame, bins: int = 10) -> float:
    work = frame.copy()
    work["bin"] = pd.cut(work["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = work.groupby("bin", observed=True).agg(
        n=("label", "size"), observed=("label", "mean"), predicted=("p", "mean")
    )
    return float(((grouped["n"] / len(work)) * (grouped["observed"] - grouped["predicted"]).abs()).sum())


def curve(frame: pd.DataFrame, paradigm: str) -> pd.DataFrame:
    work = frame.copy()
    work["bin"] = pd.cut(work["p"], np.linspace(0, 1, 11), include_lowest=True)
    result = work.groupby("bin", observed=True).agg(
        n=("label", "size"), mean_probability=("p", "mean"), observed_fraction=("label", "mean")
    ).reset_index()
    result.insert(0, "paradigm", paradigm)
    result["bin"] = result["bin"].astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    figure, axis = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Ideal")
    colors = {"TRTR": "#0072B2", "TRSTR": "#D55E00"}
    for index, paradigm in enumerate(PARADIGMS):
        frame = patient_level(ensemble_predictions(args.v6_root, f"predictions_{paradigm}.csv"))
        intercept, slope = calibration_parameters(frame)
        int_low, int_high, int_valid = stratified_bootstrap(
            frame, lambda x: calibration_parameters(x)[0], seed=6100 + index, replicates=args.bootstrap
        )
        slope_low, slope_high, slope_valid = stratified_bootstrap(
            frame, lambda x: calibration_parameters(x)[1], seed=6200 + index, replicates=args.bootstrap
        )
        rows.append(
            {
                "paradigm": paradigm,
                "n_patients": len(frame),
                "n_positive": int(frame["label"].sum()),
                "n_negative": int((1 - frame["label"]).sum()),
                "calibration_intercept": intercept,
                "intercept_ci_low": int_low,
                "intercept_ci_high": int_high,
                "calibration_slope": slope,
                "slope_ci_low": slope_low,
                "slope_ci_high": slope_high,
                "brier": float(brier_score_loss(frame["label"], frame["p"])),
                "ece_10_bins": ece(frame),
                "bootstrap_replicates_requested": args.bootstrap,
                "valid_intercept_replicates": int_valid,
                "valid_slope_replicates": slope_valid,
            }
        )
        binned = curve(frame, paradigm)
        curves.append(binned)
        axis.plot(
            binned["mean_probability"], binned["observed_fraction"], marker="o",
            color=colors[paradigm], label={"TRTR": "Real only", "TRSTR": "Diffusion augmentation"}[paradigm]
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "calibration_metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(args.out_dir / "calibration_curve.csv", index=False)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False)
    figure.savefig(args.out_dir / "calibration_curve.png", dpi=300)
    plt.close(figure)
    (args.out_dir / "status.json").write_text(json.dumps({"status": "PASS", "rows": len(result)}, indent=2) + "\n")
    print(f"CALIBRATION COMPLETE: {args.out_dir}")


if __name__ == "__main__":
    main()
