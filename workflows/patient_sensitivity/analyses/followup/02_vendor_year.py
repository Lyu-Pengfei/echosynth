"""Acquisition-period/vendor sensitivity analyses and metadata-only control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PARADIGMS, auc_ci, ensemble_predictions, patient_level


def normalized_text(series: pd.Series) -> pd.Series:
    result = series.fillna("MISSING").astype(str).str.strip()
    return result.mask(result.eq(""), "MISSING")


def eligible_metrics(frame: pd.DataFrame, seed: int, minimum_per_class: int = 10) -> dict[str, object]:
    patients = patient_level(frame)
    counts = patients["label"].value_counts()
    if counts.get(0, 0) < minimum_per_class or counts.get(1, 0) < minimum_per_class:
        return {
            "status": "INSUFFICIENT_SUPPORT",
            "n_patients": len(patients),
            "n_positive": int(counts.get(1, 0)),
            "n_negative": int(counts.get(0, 0)),
            "auroc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    point, low, high = auc_ci(patients, seed)
    return {
        "status": "ESTIMATED",
        "n_patients": len(patients),
        "n_positive": int(counts.get(1, 0)),
        "n_negative": int(counts.get(0, 0)),
        "auroc": point,
        "ci_low": low,
        "ci_high": high,
    }


def patient_metadata(manifest: pd.DataFrame) -> pd.DataFrame:
    work = manifest.copy()
    work["manufacturer"] = normalized_text(work.get("manufacturer", pd.Series(index=work.index, dtype=object)))
    work["acquisition_year"] = pd.to_numeric(work.get("acquisition_year"), errors="coerce")

    def mode_or_missing(values: pd.Series) -> str:
        modes = values.dropna().astype(str).mode()
        return modes.iloc[0] if len(modes) else "MISSING"

    return work.groupby("patient_id", as_index=False).agg(
        split=("split", "first"),
        group=("group", "first"),
        acquisition_year=("acquisition_year", "median"),
        manufacturer=("manufacturer", mode_or_missing),
        n_encounters=("encounter_id", "nunique"),
    )


def metadata_only_control(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    frame["label"] = frame["group"].astype(str).eq("CA").astype(int)
    frame["year_missing"] = frame["acquisition_year"].isna().astype(int)
    train = frame[frame["split"].astype(str).eq("train")].copy()
    mean_year = float(train["acquisition_year"].mean())
    frame["acquisition_year"] = frame["acquisition_year"].fillna(mean_year)
    design = pd.get_dummies(
        frame[["acquisition_year", "year_missing", "manufacturer"]],
        columns=["manufacturer"], dtype=float,
    )
    year_sd = float(design.loc[train.index, "acquisition_year"].std()) or 1.0
    design["acquisition_year"] = (design["acquisition_year"] - mean_year) / year_sd
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2024)
    train_index = frame["split"].astype(str).eq("train")
    model.fit(design.loc[train_index], frame.loc[train_index, "label"])
    rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        selected = frame["split"].astype(str).eq(split)
        labels = frame.loc[selected, "label"]
        probabilities = model.predict_proba(design.loc[selected])[:, 1]
        rows.append(
            {
                "split": split,
                "n_patients": int(selected.sum()),
                "n_positive": int(labels.sum()),
                "n_negative": int((1 - labels).sum()),
                "auroc": float(roc_auc_score(labels, probabilities)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.v6_root / "work/full_revision/manifests/ca_master.csv"
    manifest = pd.read_csv(manifest_path)
    required = {"encounter_id", "patient_id", "group", "split", "acquisition_year", "manufacturer"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"manifest lacks required fields: {sorted(missing)}")
    encounter_meta = manifest.groupby("encounter_id", as_index=False).agg(
        acquisition_year=("acquisition_year", "first"), manufacturer=("manufacturer", "first")
    )
    encounter_meta["acquisition_year"] = pd.to_numeric(encounter_meta["acquisition_year"], errors="coerce")
    encounter_meta["manufacturer"] = normalized_text(encounter_meta["manufacturer"])
    rows: list[dict[str, object]] = []
    for paradigm_index, paradigm in enumerate(PARADIGMS):
        predictions = ensemble_predictions(args.v6_root, f"predictions_{paradigm}.csv")
        joined = predictions.merge(encounter_meta, on="encounter_id", how="left", validate="one_to_one")
        subgroups: list[tuple[str, str, pd.DataFrame]] = [
            ("period", "2015-2023", joined[joined["acquisition_year"].between(2015, 2023)]),
            ("period", "before-2015", joined[joined["acquisition_year"] < 2015]),
            ("period", "after-2023", joined[joined["acquisition_year"] > 2023]),
        ]
        for manufacturer, subgroup in joined.groupby("manufacturer", dropna=False):
            subgroups.append(("manufacturer", str(manufacturer), subgroup))
        for subgroup_index, (kind, name, subgroup) in enumerate(subgroups):
            rows.append(
                {
                    "paradigm": paradigm,
                    "subgroup_type": kind,
                    "subgroup": name,
                    **eligible_metrics(subgroup, 7000 + 100 * paradigm_index + subgroup_index),
                }
            )
    pd.DataFrame(rows).to_csv(args.out_dir / "vendor_year_performance.csv", index=False)
    metadata = patient_metadata(manifest)
    metadata.groupby(["split", "group", "manufacturer"], dropna=False).size().rename("n_patients").reset_index().to_csv(
        args.out_dir / "vendor_distribution.csv", index=False
    )
    metadata.assign(acquisition_year=metadata["acquisition_year"].round().astype("Int64")).groupby(
        ["split", "group", "acquisition_year"], dropna=False
    ).size().rename("n_patients").reset_index().to_csv(args.out_dir / "year_distribution.csv", index=False)
    control = metadata_only_control(metadata)
    control.to_csv(args.out_dir / "vendor_year_only_control.csv", index=False)
    (args.out_dir / "status.json").write_text(
        json.dumps({"status": "PASS", "manifest": str(manifest_path), "minimum_per_class": 10}, indent=2) + "\n"
    )
    print(f"VENDOR/YEAR ANALYSIS COMPLETE: {args.out_dir}")


if __name__ == "__main__":
    main()
