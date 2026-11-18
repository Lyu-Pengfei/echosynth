"""Build the formal reviewer tables from locked multiseed test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from echosynth.eval.delong import delong_ci, delong_paired_delta_ci
from echosynth.eval.metrics import clinical_binary_metrics


def _ensemble(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, run_dir in enumerate(run_dirs):
        frame = pd.read_csv(run_dir / filename)
        keep = ["encounter_id", "patient_id", "group", "split", "label", "p"]
        frame = frame[keep].rename(columns={"p": f"p_seed{index}"})
        frames.append(frame)
    merged = frames[0]
    keys = ["encounter_id", "patient_id", "group", "split", "label"]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    probability_columns = [column for column in merged if column.startswith("p_seed")]
    merged["p"] = merged[probability_columns].mean(axis=1)
    return merged[keys + ["p"]]


def _patient_level(frame: pd.DataFrame) -> pd.DataFrame:
    consistency = frame.groupby("patient_id")["label"].nunique()
    if (consistency > 1).any():
        raise RuntimeError("a patient has conflicting labels")
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        p=("p", "mean"),
        group=("group", "first"),
        split=("split", "first"),
        n_encounters=("encounter_id", "nunique"),
    )


def _bootstrap_auprc(frame: pd.DataFrame, seed: int, n_bootstrap: int = 2000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    patients = frame["patient_id"].drop_duplicates().to_numpy()
    grouped = {key: value for key, value in frame.groupby("patient_id", sort=False)}
    values: list[float] = []
    attempts = 0
    while len(values) < n_bootstrap and attempts < n_bootstrap * 3:
        attempts += 1
        sampled = generator.choice(patients, size=len(patients), replace=True)
        labels = np.concatenate([grouped[value]["label"].to_numpy() for value in sampled])
        scores = np.concatenate([grouped[value]["p"].to_numpy() for value in sampled])
        if len(np.unique(labels)) == 2:
            values.append(float(average_precision_score(labels, scores)))
    if len(values) < n_bootstrap // 2:
        raise RuntimeError("too few valid patient-cluster bootstrap replicates")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _target_sensitivity_threshold(validation: pd.DataFrame, target: float = 0.90) -> float:
    positives = validation.loc[validation["label"].astype(int) == 1, "p"].to_numpy()
    if len(positives) < 2:
        raise RuntimeError("validation set has too few positive patients")
    return float(np.quantile(positives, 1.0 - target, method="lower"))


def _metrics(
    frame: pd.DataFrame,
    *,
    threshold: float,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    auc, low, high = delong_ci(frame["label"], frame["p"])
    auprc = float(average_precision_score(frame["label"], frame["p"]))
    pr_low, pr_high = _bootstrap_auprc(frame, seed, n_bootstrap)
    return {
        "n_patients": int(frame["patient_id"].nunique()),
        "n_encounters": int(len(frame)),
        "auroc": auc,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "auprc": auprc,
        "auprc_ci_low": pr_low,
        "auprc_ci_high": pr_high,
        **clinical_binary_metrics(frame["label"], frame["p"], threshold=threshold),
    }


def _calibration(frame: pd.DataFrame, paradigm: str, unit: str) -> pd.DataFrame:
    work = frame.copy()
    work["bin"] = pd.cut(work["p"], np.linspace(0, 1, 11), include_lowest=True)
    output = work.groupby("bin", observed=True).agg(
        n=("label", "size"), mean_probability=("p", "mean"), observed_fraction=("label", "mean")
    ).reset_index()
    output.insert(0, "unit", unit)
    output.insert(0, "paradigm", paradigm)
    output["bin"] = output["bin"].astype(str)
    return output


def _decision_curve(frame: pd.DataFrame, paradigm: str, unit: str) -> pd.DataFrame:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["p"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        predictions = scores >= threshold
        tp = int(((labels == 1) & predictions).sum())
        fp = int(((labels == 0) & predictions).sum())
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "paradigm": paradigm,
                "unit": unit,
                "threshold": float(threshold),
                "net_benefit": float(tp / len(labels) - fp / len(labels) * odds),
                "treat_all_net_benefit": float(labels.mean() - (1 - labels.mean()) * odds),
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _safe_auc(frame: pd.DataFrame) -> dict[str, Any] | None:
    if len(frame) < 4 or frame["label"].nunique() != 2 or frame["label"].value_counts().min() < 2:
        return None
    auc, low, high = delong_ci(frame["label"], frame["p"])
    return {"n": len(frame), "auroc": auc, "ci_low": low, "ci_high": high}


def _reviewer_availability(metadata: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Unique patient identifier", "available", 1.0, "DICOM (0010,0020), hashed before output"],
            ["Patient-level partition", "available", 1.0, "All encounters from one DICOM PatientID stay in one split"],
            ["Acquisition year", "available", metadata["acquisition_year_completeness"], "DICOM StudyDate (0008,0020)"],
            ["Scanner manufacturer", "available", metadata["manufacturer_completeness"], "DICOM Manufacturer (0008,0070)"],
            ["Scanner model", "available_if_present", metadata["manufacturer_model_completeness"], "DICOM ManufacturerModelName (0008,1090)"],
            ["Standardized view", "limited", metadata["standardized_view_completeness"], "ViewCodeSequence only; no inferred labels without validation"],
            ["Standardized cardiac phase", "not_available", metadata["standardized_cardiac_phase_completeness"], "No validated ED/MS/ES label in current archive"],
            ["LVH quantitative threshold", "pending_clinical_confirmation", 0.0, "IVSd and LVPWd per ASE; exact numeric/sex-specific criteria still needed"],
            ["Occult CA exclusion in LVH controls", "known_limitation", 1.0, "Not universally excluded with confirmatory testing"],
            ["CA subtype and LVEF", "pending_database_owner", 0.0, "Expected in Mike Khouri database; linkage not supplied"],
            ["CA diagnostic reference standard", "narrative_available", 1.0, "Expert diagnosis using clinical testing and guidelines, not one test"],
            ["Unified CA disease severity", "possibly_unavailable", 0.0, "A standardized single severity variable may not exist"],
            ["Sex/age/race fairness", "requires_clinical_linkage", 0.0, "Not present in the image archive"],
            ["Original survey denominator audit", "requires_survey_export", 0.0, "Original respondent-level survey data not provided"],
            ["EchoNet-LVH partition audit", "requires_external_manifest", 0.0, "EchoNet-LVH manifests are outside this Duke archive"],
        ],
        columns=["reviewer_data_question", "status", "completeness_fraction", "evidence_or_next_step"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_prcc_full.yaml"))
    parser.add_argument("--work-root", type=Path, default=Path("work/full_revision"))
    parser.add_argument("--results-root", type=Path, default=Path("results/full_revision"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["analysis"]["seeds"]]
    n_bootstrap = int(config["analysis"].get("bootstrap_replicates", 2000))
    run_dirs = [args.results_root / "runs" / f"ca_revision_full_seed{seed}" for seed in seeds]
    missing = [str(path) for path in run_dirs if not (path / "metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed seed runs: {missing}")
    report_dir = args.results_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    paradigms = [
        "TRTR",
        "TRSTR",
        "BOOTSTRAP",
        "CLASS_WEIGHTED",
        "BALANCED_SAMPLING",
        "TSTR",
    ]
    summary_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    ensemble: dict[str, pd.DataFrame] = {}
    for index, paradigm in enumerate(paradigms):
        test = _ensemble(run_dirs, f"predictions_{paradigm}.csv")
        validation = _ensemble(run_dirs, f"validation_predictions_{paradigm}.csv")
        ensemble[paradigm] = test
        for unit, test_unit, validation_unit in (
            ("encounter", test, validation),
            ("patient", _patient_level(test), _patient_level(validation)),
        ):
            threshold = _target_sensitivity_threshold(validation_unit, 0.90)
            summary_rows.append(
                {
                    "endpoint": "composed_CA_vs_all",
                    "paradigm": paradigm,
                    "unit": unit,
                    "validation_target_sensitivity": 0.90,
                    "selected_threshold": threshold,
                    **_metrics(
                        test_unit,
                        threshold=threshold,
                        seed=2024 + index,
                        n_bootstrap=n_bootstrap,
                    ),
                }
            )
            calibration_frames.append(_calibration(test_unit, paradigm, unit))
            decision_frames.append(_decision_curve(test_unit, paradigm, unit))
        stage2_test = _ensemble(run_dirs, f"predictions_stage2_{paradigm}.csv")
        stage2_validation = _ensemble(
            run_dirs, f"validation_predictions_stage2_{paradigm}.csv"
        )
        for unit, test_unit, validation_unit in (
            ("encounter", stage2_test, stage2_validation),
            ("patient", _patient_level(stage2_test), _patient_level(stage2_validation)),
        ):
            threshold = _target_sensitivity_threshold(validation_unit, 0.90)
            summary_rows.append(
                {
                    "endpoint": "stage2_CA_vs_LVH",
                    "paradigm": paradigm,
                    "unit": unit,
                    "validation_target_sensitivity": 0.90,
                    "selected_threshold": threshold,
                    **_metrics(
                        test_unit,
                        threshold=threshold,
                        seed=3024 + index,
                        n_bootstrap=n_bootstrap,
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "primary_performance.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        report_dir / "calibration_curves.csv", index=False
    )
    pd.concat(decision_frames, ignore_index=True).to_csv(
        report_dir / "decision_curves.csv", index=False
    )

    delta_rows: list[dict[str, Any]] = []
    baseline = ensemble["TRTR"]
    for paradigm, frame in ensemble.items():
        if paradigm == "TRTR":
            continue
        for unit, first, second, key in (
            ("encounter", baseline, frame, "encounter_id"),
            ("patient", _patient_level(baseline), _patient_level(frame), "patient_id"),
        ):
            merged = first[[key, "label", "p"]].merge(
                second[[key, "p"]], on=key, suffixes=("_trtr", "_arm")
            )
            result = delong_paired_delta_ci(
                merged["label"], merged["p_trtr"], merged["p_arm"]
            )
            delta_rows.append({"paradigm": paradigm, "unit": unit, **result.__dict__})
    pd.DataFrame(delta_rows).to_csv(report_dir / "paired_auroc_differences.csv", index=False)

    manifest = pd.read_csv(args.work_root / "manifests" / "ca_master.csv")
    encounter_meta = manifest.groupby("encounter_id", as_index=False).agg(
        acquisition_year=("acquisition_year", "first"),
        manufacturer=("manufacturer", "first"),
        group=("group", "first"),
    )
    year_ranges = manifest.dropna(subset=["acquisition_year"]).groupby("group")[
        "acquisition_year"
    ].agg(["min", "max"])
    common_start = int(year_ranges["min"].max())
    common_end = int(year_ranges["max"].min())
    subgroup_rows: list[dict[str, Any]] = []
    for paradigm, predictions in ensemble.items():
        joined = predictions.merge(encounter_meta, on="encounter_id", suffixes=("", "_meta"))
        common = joined[
            joined["acquisition_year"].between(common_start, common_end, inclusive="both")
        ]
        result = _safe_auc(common)
        if result:
            subgroup_rows.append(
                {
                    "paradigm": paradigm,
                    "subgroup_type": "common_acquisition_period",
                    "subgroup": f"{common_start}-{common_end}",
                    **result,
                }
            )
        for manufacturer, subgroup in joined.groupby("manufacturer", dropna=False):
            result = _safe_auc(subgroup)
            if result:
                subgroup_rows.append(
                    {
                        "paradigm": paradigm,
                        "subgroup_type": "manufacturer",
                        "subgroup": str(manufacturer) or "missing",
                        **result,
                    }
                )
    pd.DataFrame(subgroup_rows).to_csv(report_dir / "year_vendor_subgroups.csv", index=False)
    year_ranges.reset_index().to_csv(report_dir / "acquisition_year_ranges.csv", index=False)

    seed_rows: list[dict[str, Any]] = []
    for seed, run_dir in zip(seeds, run_dirs, strict=True):
        payload = json.loads((run_dir / "metrics.json").read_text())
        for paradigm in paradigms:
            block = payload[paradigm]
            seed_rows.append(
                {
                    "seed": seed,
                    "paradigm": paradigm,
                    "selected_synthetic_count": payload["selected_synthetic_count"],
                    "auroc": block["auroc"],
                    "auprc": block["auprc"],
                    "brier": block["brier"],
                    "stage2_auroc": block["stage2_ca_vs_lvh"]["auroc"],
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(report_dir / "seed_level_results.csv", index=False)
    seed_frame.groupby("paradigm").agg(
        n_seeds=("seed", "nunique"),
        auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_sd=("auprc", "std"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
        stage2_auroc_mean=("stage2_auroc", "mean"),
        stage2_auroc_sd=("stage2_auroc", "std"),
    ).reset_index().to_csv(report_dir / "training_variability.csv", index=False)

    sensitivity_rows: list[dict[str, Any]] = []
    for name in ("cap5", "resolution224"):
        path = args.results_root / "runs" / f"sensitivity_{name}_seed2024" / "metrics.json"
        if path.is_file():
            block = json.loads(path.read_text())["TRTR"]
            sensitivity_rows.append(
                {"analysis": name, "paradigm": "TRTR", "auroc": block["auroc"], "auprc": block["auprc"], "brier": block["brier"]}
            )
    pd.DataFrame(sensitivity_rows).to_csv(report_dir / "sensitivity_analyses.csv", index=False)

    metadata = json.loads((args.work_root / "metadata_completeness.json").read_text())
    availability = _reviewer_availability(metadata)
    availability.to_csv(report_dir / "reviewer_data_availability.csv", index=False)
    availability.rename(
        columns={"reviewer_data_question": "reviewer_request", "evidence_or_next_step": "response_evidence"}
    ).to_csv(report_dir / "reviewer_question_matrix.csv", index=False)
    methods = {
        "analysis_status": "formal multiseed test analysis",
        "seeds": seeds,
        "split": metadata["split_policy"],
        "image_rule": "first decoded frame from each eligible grayscale DICOM",
        "primary_resolution": 112,
        "sensitivity_resolution": 224,
        "synthetic_count_selection": "validation AUROC only; grid 500/1000/1500/2000; held-out test accessed after selection",
        "test_units": ["encounter", "patient"],
        "auroc_interval": "DeLong 95% CI",
        "auprc_interval": "2000-replicate patient-cluster bootstrap 95% CI",
        "paired_difference": "paired DeLong test on locked encounters",
        "operating_point": "threshold selected on validation for 90% sensitivity",
        "comparators": paradigms,
        "pending_noncomputable_items": availability.loc[
            ~availability["status"].isin(["available", "narrative_available", "known_limitation"]),
            "reviewer_data_question",
        ].tolist(),
    }
    (report_dir / "methods_and_provenance.json").write_text(json.dumps(methods, indent=2) + "\n")
    best = summary[(summary["endpoint"] == "composed_CA_vs_all") & (summary["unit"] == "patient")]
    lines = [
        "# Formal AML/CA revision analysis",
        "",
        f"Patient leakage count: {metadata['patient_leakage_count']}",
        f"Common acquisition period: {common_start}-{common_end}",
        "",
        "## Patient-level primary results",
        "",
        best[["paradigm", "n_patients", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "brier"]].to_csv(index=False),
        "",
        "Items requiring clinical/external data are listed in reviewer_data_availability.csv.",
    ]
    (report_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"FORMAL REPORT PASSED out={report_dir}")


if __name__ == "__main__":
    main()
