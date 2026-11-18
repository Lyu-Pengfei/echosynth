#!/usr/bin/env python3
"""Lock the formal v6 patient-level operating-point results.

This script mirrors the v6 formal-report rules:
1. average probabilities across completed seeds at the encounter level;
2. average repeat encounters within patient;
3. select each arm's threshold on validation patients to target 90% sensitivity;
4. apply the locked threshold to held-out test patients;
5. estimate the paired specificity difference by paired patient resampling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {"TRTR": "real_only", "TRSTR": "diffusion_augmentation"}
KEYS = ["encounter_id", "patient_id", "group", "split", "label"]


def discover_runs(results_root: Path) -> list[Path]:
    runs = sorted((results_root / "runs").glob("ca_revision_full_seed*"))
    valid = [run for run in runs if all((run / f"predictions_{arm}.csv").is_file() for arm in ARMS)]
    if len(valid) < 2:
        raise FileNotFoundError("Fewer than two completed ca_revision_full_seed* runs were found")
    return valid


def ensemble(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for index, run_dir in enumerate(run_dirs):
        frame = pd.read_csv(run_dir / filename)
        missing = set(KEYS + ["p"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{run_dir / filename} is missing columns: {sorted(missing)}")
        frame = frame[KEYS + ["p"]].rename(columns={"p": f"p_seed{index}"})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(frames[0]):
        raise RuntimeError(f"Seed files do not contain identical rows for {filename}")
    probability_columns = [column for column in merged if column.startswith("p_seed")]
    merged["p"] = merged[probability_columns].mean(axis=1)
    return merged[KEYS + ["p"]]


def patient_level(frame: pd.DataFrame) -> pd.DataFrame:
    if (frame.groupby("patient_id")["label"].nunique() > 1).any():
        raise RuntimeError("A patient has conflicting labels")
    if (frame.groupby("patient_id")["split"].nunique() > 1).any():
        raise RuntimeError("A patient spans multiple partitions")
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        p=("p", "mean"),
        group=("group", "first"),
        split=("split", "first"),
        n_encounters=("encounter_id", "nunique"),
    )


def target_sensitivity_threshold(validation: pd.DataFrame, target: float = 0.90) -> float:
    positives = validation.loc[validation["label"].astype(int) == 1, "p"].to_numpy(float)
    if len(positives) < 2:
        raise RuntimeError("Validation set has fewer than two positive patients")
    return float(np.quantile(positives, 1.0 - target, method="lower"))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def operating_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(int)
    predicted = frame["p"].to_numpy(float) >= threshold
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    sens_low, sens_high = wilson(tp, tp + fn)
    spec_low, spec_high = wilson(tn, tn + fp)
    ppv_low, ppv_high = wilson(tp, tp + fp)
    npv_low, npv_high = wilson(tn, tn + fn)
    return {
        "n_patients": int(len(frame)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "sensitivity_ci_low": sens_low,
        "sensitivity_ci_high": sens_high,
        "specificity": specificity,
        "specificity_ci_low": spec_low,
        "specificity_ci_high": spec_high,
        "ppv": ppv,
        "ppv_ci_low": ppv_low,
        "ppv_ci_high": ppv_high,
        "npv": npv,
        "npv_ci_low": npv_low,
        "npv_ci_high": npv_high,
    }


def decision_curve(frame: pd.DataFrame, arm: str, thresholds: tuple[float, ...]) -> list[dict[str, float | int | str]]:
    labels = frame["label"].to_numpy(int)
    scores = frame["p"].to_numpy(float)
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "arm": arm,
                "threshold_probability": threshold,
                "tp": tp,
                "fp": fp,
                "net_benefit": tp / len(labels) - fp / len(labels) * odds,
                "treat_all_net_benefit": labels.mean() - (1.0 - labels.mean()) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return rows


def paired_specificity_interval(
    real: pd.DataFrame,
    augmented: pd.DataFrame,
    real_threshold: float,
    augmented_threshold: float,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    paired = real[["patient_id", "label", "p"]].merge(
        augmented[["patient_id", "label", "p"]],
        on=["patient_id", "label"],
        suffixes=("_real", "_augmented"),
        validate="one_to_one",
    )
    if len(paired) != len(real) or len(paired) != len(augmented):
        raise RuntimeError("Real-only and augmented test patients are not identical")
    negatives = paired.loc[paired["label"].astype(int) == 0].reset_index(drop=True)
    real_correct = (negatives["p_real"].to_numpy(float) < real_threshold).astype(float)
    augmented_correct = (negatives["p_augmented"].to_numpy(float) < augmented_threshold).astype(float)
    observed_real = float(real_correct.mean())
    observed_augmented = float(augmented_correct.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(negatives), size=(n_resamples, len(negatives)))
    deltas = augmented_correct[draws].mean(axis=1) - real_correct[draws].mean(axis=1)
    return {
        "contrast": "diffusion_augmentation_minus_real_only",
        "n_negative_patients": int(len(negatives)),
        "real_only_specificity": observed_real,
        "diffusion_augmentation_specificity": observed_augmented,
        "specificity_difference": observed_augmented - observed_real,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "resampling_unit": "paired negative patient",
        "n_resamples": int(n_resamples),
        "random_seed": int(seed),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/full_revision"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "report" / "final_operating_point"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_runs(args.results_root)

    validation: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, float] = {}
    metrics_rows = []
    decision_rows = []
    for paradigm, arm in ARMS.items():
        validation[arm] = patient_level(ensemble(run_dirs, f"validation_predictions_{paradigm}.csv"))
        test[arm] = patient_level(ensemble(run_dirs, f"predictions_{paradigm}.csv"))
        thresholds[arm] = target_sensitivity_threshold(validation[arm], 0.90)
        metrics_rows.append({"arm": arm, **operating_metrics(test[arm], thresholds[arm])})
        decision_rows.extend(decision_curve(test[arm], arm, (0.05, 0.10)))

    metrics = pd.DataFrame(metrics_rows)
    decisions = pd.DataFrame(decision_rows)
    paired = pd.DataFrame(
        [
            paired_specificity_interval(
                test["real_only"],
                test["diffusion_augmentation"],
                thresholds["real_only"],
                thresholds["diffusion_augmentation"],
                n_resamples=args.resamples,
                seed=args.seed,
            )
        ]
    )
    metrics.to_csv(output_dir / "operating_point_metrics.csv", index=False)
    decisions.to_csv(output_dir / "decision_curve_5_10.csv", index=False)
    paired.to_csv(output_dir / "paired_specificity_difference.csv", index=False)

    lines = [
        "# Formal v6 operating-point lock",
        "",
        f"Completed seed runs: {len(run_dirs)}",
        "Threshold rule: five-seed ensemble, patient-level validation threshold targeting 90% sensitivity.",
        "",
    ]
    for row in metrics.to_dict("records"):
        lines.extend(
            [
                f"## {row['arm']}",
                "",
                f"- Threshold: {row['threshold']:.6f}",
                f"- Test patients: {row['n_patients']} ({row['n_positive']} positive, {row['n_negative']} negative)",
                f"- TP/FP/TN/FN: {row['tp']}/{row['fp']}/{row['tn']}/{row['fn']}",
                f"- Sensitivity: {fmt(row['sensitivity'])} (95% CI {fmt(row['sensitivity_ci_low'])}-{fmt(row['sensitivity_ci_high'])})",
                f"- Specificity: {fmt(row['specificity'])} (95% CI {fmt(row['specificity_ci_low'])}-{fmt(row['specificity_ci_high'])})",
                f"- PPV: {fmt(row['ppv'])} (95% CI {fmt(row['ppv_ci_low'])}-{fmt(row['ppv_ci_high'])})",
                f"- NPV: {fmt(row['npv'])} (95% CI {fmt(row['npv_ci_low'])}-{fmt(row['npv_ci_high'])})",
                "",
            ]
        )
    difference = paired.iloc[0]
    lines.extend(
        [
            "## Paired specificity contrast",
            "",
            f"- Difference (augmentation - real only): {fmt(difference['specificity_difference'])}",
            f"- Paired 95% interval: {fmt(difference['ci_low'])} to {fmt(difference['ci_high'])}",
            "",
            "## Decision curves at 5% and 10%",
            "",
            decisions.to_csv(index=False),
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"FORMAL OPERATING-POINT ANALYSIS PASSED: {output_dir}")


if __name__ == "__main__":
    main()
