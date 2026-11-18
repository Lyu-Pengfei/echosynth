"""Patient-level acquisition-era and vendor-adjusted sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_utils import (
    auc_summary,
    load_config,
    normalize_vendor,
    paired_auc_delta,
    patient_metadata,
    project_path,
    write_json,
)


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        raise ValueError(f"cannot identify seed from {path}")
    return int(match.group(1))


def load_arm_predictions(project: Path, seeds: list[int], arm: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        path = (
            project
            / "results/full_revision/runs"
            / f"ca_revision_full_seed{seed}"
            / f"predictions_{arm}.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"patient_id", "label", "p"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        patient = frame.groupby("patient_id", observed=True).agg(
            label=("label", "first"), p=("p", "mean")
        ).reset_index()
        patient["seed"] = seed
        rows.append(patient)
    combined = pd.concat(rows, ignore_index=True)
    consistency = combined.groupby("patient_id")["label"].nunique()
    if (consistency > 1).any():
        raise ValueError("prediction labels differ across seeds")
    return combined


def ensemble_predictions(project: Path, seeds: list[int]) -> pd.DataFrame:
    real = load_arm_predictions(project, seeds, "TRTR")
    diffusion = load_arm_predictions(project, seeds, "TRSTR")
    real_wide = real.pivot(index=["patient_id", "label"], columns="seed", values="p")
    diff_wide = diffusion.pivot(index=["patient_id", "label"], columns="seed", values="p")
    common = real_wide.index.intersection(diff_wide.index)
    if len(common) != len(real_wide) or len(common) != len(diff_wide):
        raise ValueError("TRTR and TRSTR patient sets differ")
    output = real_wide.loc[common].mean(axis=1).rename("p_real").to_frame()
    output["p_diffusion"] = diff_wide.loc[common].mean(axis=1)
    return output.reset_index()


def oof_probabilities(frame: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    labels = frame["label"].astype(int).to_numpy()
    minimum_class = int(np.bincount(labels).min())
    folds = min(5, minimum_class)
    if folds < 3:
        raise ValueError("too few cases for acquisition-adjusted cross-validation")
    categorical = [column for column in features if column == "vendor"]
    numeric = [column for column in features if column != "vendor"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    model = Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]
    )
    predictions = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(frame, labels):
        model.fit(frame.iloc[train_index][features], labels[train_index])
        predictions[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
    return predictions


def subgroup_row(
    frame: pd.DataFrame,
    name: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"subgroup": name}
    if len(frame) < 10 or frame["label"].nunique() < 2:
        result.update(
            {
                "status": "INSUFFICIENT",
                "n": len(frame),
                "n_positive": int(frame["label"].sum()),
                "n_negative": int((1 - frame["label"]).sum()),
            }
        )
        return result
    result["status"] = "ESTIMATED"
    for arm, column in (("real", "p_real"), ("diffusion", "p_diffusion")):
        metrics = auc_summary(frame, column, replicates, seed + (arm == "diffusion"))
        for key, value in metrics.items():
            result[f"{arm}_{key}"] = value
    result.update(
        paired_auc_delta(frame, "p_real", "p_diffusion", replicates, seed + 7)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["analysis"]
    project = Path(config["project_root"])
    manifest_path = project_path(config, config["inputs"]["real_manifest"])
    manifest = pd.read_csv(manifest_path)
    metadata = patient_metadata(manifest)
    predictions = ensemble_predictions(project, [int(seed) for seed in settings["seeds"]])
    frame = predictions.merge(metadata, on=["patient_id", "label"], how="left", validate="one_to_one")
    if frame[["acquisition_year", "vendor"]].isna().all(axis=None):
        raise RuntimeError("acquisition metadata did not link to predictions")
    frame["vendor"] = frame["vendor"].map(normalize_vendor)
    frame["p_real_logit"] = np.log(
        np.clip(frame["p_real"], 1e-6, 1 - 1e-6)
        / np.clip(1 - frame["p_real"], 1e-6, 1 - 1e-6)
    )
    frame["p_diffusion_logit"] = np.log(
        np.clip(frame["p_diffusion"], 1e-6, 1 - 1e-6)
        / np.clip(1 - frame["p_diffusion"], 1e-6, 1 - 1e-6)
    )
    frame["p_acquisition_only"] = oof_probabilities(
        frame, ["acquisition_year", "vendor"], int(settings["seeds"][0])
    )
    frame["p_real_adjusted"] = oof_probabilities(
        frame, ["p_real_logit", "acquisition_year", "vendor"], int(settings["seeds"][0]) + 1
    )
    frame["p_diffusion_adjusted"] = oof_probabilities(
        frame,
        ["p_diffusion_logit", "acquisition_year", "vendor"],
        int(settings["seeds"][0]) + 2,
    )

    replicates = int(settings["bootstrap_replicates"])
    start = int(settings["common_year_start"])
    end = int(settings["common_year_end"])
    common_era = frame[frame["acquisition_year"].between(start, end, inclusive="both")].copy()
    vendor_counts = frame.groupby("vendor")["label"].agg(["sum", "count"])
    vendor_counts["controls"] = vendor_counts["count"] - vendor_counts["sum"]
    eligible_vendors = vendor_counts[
        (vendor_counts["sum"] >= int(settings["minimum_vendor_cases"]))
        & (vendor_counts["controls"] >= int(settings["minimum_vendor_controls"]))
    ].index.tolist()
    vendor_common = frame[frame["vendor"].isin(eligible_vendors)].copy()

    subgroup_rows = [
        subgroup_row(frame, "all_test_patients", replicates, 1101),
        subgroup_row(common_era, f"common_era_{start}_{end}", replicates, 1102),
        subgroup_row(vendor_common, "vendors_with_both_groups", replicates, 1103),
    ]
    for vendor in sorted(frame["vendor"].unique()):
        subgroup_rows.append(
            subgroup_row(frame[frame["vendor"].eq(vendor)], f"vendor_{vendor}", replicates, 1200)
        )
    for name, low, high in (
        ("era_before_2015", -np.inf, 2014),
        ("era_2015_2018", 2015, 2018),
        ("era_2019_2023", 2019, 2023),
        ("era_2024", 2024, 2024),
    ):
        subgroup_rows.append(
            subgroup_row(
                frame[frame["acquisition_year"].between(low, high, inclusive="both")],
                name,
                replicates,
                1300,
            )
        )
    subgroup = pd.DataFrame(subgroup_rows)
    out_dir = Path(__file__).resolve().parent / "results/era_vendor"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "patient_predictions_with_acquisition.csv", index=False)
    subgroup.to_csv(out_dir / "subgroup_metrics.csv", index=False)
    vendor_counts.reset_index().to_csv(out_dir / "vendor_counts.csv", index=False)

    summary = {
        "status": "PASS",
        "n_patients": len(frame),
        "n_positive": int(frame["label"].sum()),
        "n_negative": int((1 - frame["label"]).sum()),
        "year_missing": int(frame["acquisition_year"].isna().sum()),
        "vendor_unknown": int(frame["vendor"].eq("UNKNOWN").sum()),
        "common_era": {"start": start, "end": end, "n": len(common_era)},
        "eligible_vendors": eligible_vendors,
        "acquisition_only_negative_control": auc_summary(
            frame, "p_acquisition_only", replicates, 1401
        ),
        "real_only_raw": auc_summary(frame, "p_real", replicates, 1402),
        "diffusion_raw": auc_summary(frame, "p_diffusion", replicates, 1403),
        "real_plus_acquisition_oof": auc_summary(frame, "p_real_adjusted", replicates, 1404),
        "diffusion_plus_acquisition_oof": auc_summary(
            frame, "p_diffusion_adjusted", replicates, 1405
        ),
        "raw_paired_delta": paired_auc_delta(
            frame, "p_real", "p_diffusion", replicates, 1406
        ),
        "interpretation_guardrail": (
            "The acquisition-adjusted models are exploratory out-of-fold sensitivity analyses. "
            "They do not replace the prespecified held-out-real primary endpoint."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"ERA/VENDOR ANALYSIS COMPLETE: {out_dir}")


if __name__ == "__main__":
    main()
