"""Summarize five-seed no-flip results against the original v6 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import (
    PARADIGMS,
    SEEDS,
    auc_ci,
    ensemble_from_dirs,
    ensemble_predictions,
    paired_delta_ci,
    patient_level,
)


def metric_row(frame: pd.DataFrame, label: str, paradigm: str, seed: int) -> dict[str, object]:
    patient = patient_level(frame)
    auc, low, high = auc_ci(patient, seed)
    return {
        "augmentation_setting": label,
        "paradigm": paradigm,
        "n_patients": len(patient),
        "n_positive": int(patient["label"].sum()),
        "n_negative": int((1 - patient["label"]).sum()),
        "auroc": auc,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "auprc": float(average_precision_score(patient["label"], patient["p"])),
        "brier": float(brier_score_loss(patient["label"], patient["p"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--follow-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help="Completed no-flip seeds to ensemble; use one seed for the expedited sensitivity analysis.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    no_flip_dirs = [
        args.follow_root / "results/no_flip/runs" / f"ca_revision_no_flip_seed{seed}"
        for seed in args.seeds
    ]
    missing = [str(path) for path in no_flip_dirs if not (path / "metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing no-flip seed runs: {missing}")
    rows: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for index, paradigm in enumerate(PARADIGMS):
        original = ensemble_predictions(args.v6_root, f"predictions_{paradigm}.csv")
        no_flip = ensemble_from_dirs(no_flip_dirs, f"predictions_{paradigm}.csv")
        rows.append(metric_row(original, "original_horizontal_flip", paradigm, 9000 + index))
        rows.append(metric_row(no_flip, "no_horizontal_flip", paradigm, 9010 + index))
        delta, low, high = paired_delta_ci(
            patient_level(original), patient_level(no_flip), seed=9020 + index
        )
        deltas.append(
            {
                "paradigm": paradigm,
                "delta_no_flip_minus_original": delta,
                "ci_low": low,
                "ci_high": high,
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "no_flip_performance.csv", index=False)
    pd.DataFrame(deltas).to_csv(args.out_dir / "no_flip_paired_differences.csv", index=False)
    selected = []
    for seed, directory in zip(args.seeds, no_flip_dirs, strict=True):
        payload = json.loads((directory / "metrics.json").read_text())
        selected.append({"seed": seed, "selected_synthetic_count": payload["selected_synthetic_count"]})
    pd.DataFrame(selected).to_csv(args.out_dir / "no_flip_selected_counts.csv", index=False)
    (args.out_dir / "analysis_scope.json").write_text(
        json.dumps(
            {
                "analysis": "horizontal-flip removal sensitivity analysis",
                "seeds": args.seeds,
                "n_seeds": len(args.seeds),
                "interpretation": "Expedited single-seed sensitivity analysis" if len(args.seeds) == 1 else "Multi-seed sensitivity analysis",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"NO-FLIP REPORT COMPLETE: {args.out_dir}")


if __name__ == "__main__":
    main()
