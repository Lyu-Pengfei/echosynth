"""Aim 1 HCM cohort linkage and de-identified DICOM ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from echosynth.data.dicom_ingest import _decode, _png_bytes
from echosynth.data.manifest import ManifestSchema, write_manifest
from echosynth.utils.io import atomic_write_bytes, atomic_write_json

ROLE_TO_GROUP = {"hcm": "HCM", "lvh": "LVH", "nolvh": "GEN"}
SPLIT_MAP = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _dates(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    return values.dt.strftime("%Y-%m-%d").astype("string")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "t", "1", "yes"})


def link_cohort_inventory(
    cohort: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link downloaded Aim 1 subjects to echo studies by MRN and calendar date."""
    cohort = cohort.rename(
        columns={"echo_date": "echodate", "echo_downloaded": "echodownloaded"}
    ).copy()
    required_cohort = {"mrn", "role", "echodate", "echodownloaded", "strat"}
    required_inventory = {"mrn", "study_id", "study_date", "path"}
    missing_cohort = sorted(required_cohort - set(cohort))
    missing_inventory = sorted(required_inventory - set(inventory))
    if missing_cohort or missing_inventory:
        raise ValueError(
            f"missing columns: cohort={missing_cohort}, inventory={missing_inventory}"
        )
    cohort["role"] = cohort["role"].astype("string").str.strip().str.lower()
    cohort = cohort.loc[cohort["role"].isin(ROLE_TO_GROUP)].copy()
    cohort = cohort.loc[_truthy(cohort["echodownloaded"])].copy()
    cohort["link_mrn"] = _clean_id(cohort["mrn"])
    cohort["link_date"] = _dates(cohort["echodate"])
    cohort["group"] = cohort["role"].map(ROLE_TO_GROUP)
    cohort["split"] = cohort["strat"].astype("string").str.lower().map(SPLIT_MAP)
    if cohort["split"].isna().any():
        values = sorted(cohort.loc[cohort["split"].isna(), "strat"].astype(str).unique())
        raise ValueError(f"unrecognized strat values: {values}")
    inventory = inventory.copy()
    inventory["link_mrn"] = _clean_id(inventory["mrn"])
    inventory["link_date"] = _dates(inventory["study_date"])
    linked = cohort.merge(
        inventory, on=["link_mrn", "link_date"], how="inner", suffixes=("", "_inv")
    )
    patient_split = linked.groupby("link_mrn")["split"].nunique()
    if (patient_split > 1).any():
        raise ValueError("a patient spans more than one Lydia split")
    matched_keys = linked[["link_mrn", "link_date"]].drop_duplicates()
    eligible_keys = cohort[["link_mrn", "link_date"]].drop_duplicates()
    report: dict[str, Any] = {
        "eligible_subject_dates": int(len(eligible_keys)),
        "matched_subject_dates": int(len(matched_keys)),
        "unmatched_subject_dates": int(len(eligible_keys) - len(matched_keys)),
        "matched_studies": int(len(linked)),
        "matched_by_role": {
            str(key): int(value) for key, value in linked.groupby("role").size().items()
        },
        "note": "ATTR is excluded by design for HCM plan 1.",
    }
    keep = ["link_mrn", "link_date", "study_id", "path", "group", "split", "role"]
    return linked[keep].copy(), report


def ingest_hcm_studies(
    cohort_csv: Path,
    inventory_csv: Path,
    out_dir: Path,
    manifest_out: Path,
    train_manifest_out: Path,
    report_out: Path,
    config_path: Path,
    *,
    max_images_per_study: int = 0,
) -> pd.DataFrame:
    """Decode linked studies, discard PHI, and preserve Lydia's supplied partition."""
    cohort = pd.read_csv(cohort_csv, dtype={"mrn": "string"})
    inventory = pd.read_csv(inventory_csv, dtype={"mrn": "string"})
    linked, report = link_cohort_inventory(cohort, inventory)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed_uids = config.get("standard_sop_class_uids", [])
    if not allowed_uids:
        raise ValueError("standard_sop_class_uids is empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejection_counts: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for study in linked.itertuples(index=False):
        source = Path(str(study.path))
        files = (
            [source]
            if source.is_file()
            else sorted(path for path in source.rglob("*") if path.is_file())
        )
        if max_images_per_study > 0:
            files = files[:max_images_per_study]
        for source_file in files:
            decoded = _decode(source_file, allowed_uids)
            if decoded.reason is not None or decoded.row is None or decoded.pixels is None:
                reason = decoded.reason or "UNKNOWN"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            image_id = str(decoded.row["image_id"])
            if image_id in seen:
                rejection_counts["DUPLICATE_ID"] = rejection_counts.get("DUPLICATE_ID", 0) + 1
                continue
            seen.add(image_id)
            destination = out_dir / f"{image_id}.png"
            atomic_write_bytes(destination, _png_bytes(decoded.pixels))
            row = decoded.row
            row.update(
                {
                    "patient_id": _hash(str(study.link_mrn)),
                    "encounter_id": _hash(f"{study.link_mrn}:{study.study_id}"),
                    "group": str(study.group),
                    "study_date": str(study.link_date),
                    "dicom_path": "",
                    "png_path": str(destination),
                    "split": str(study.split),
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows, columns=ManifestSchema.columns)
    if frame.empty:
        raise RuntimeError("no DICOM images were decoded from linked studies")
    write_manifest(frame, manifest_out)
    write_manifest(frame.loc[frame["split"].astype(str) == "train"].copy(), train_manifest_out)
    report["decoded_images"] = int(len(frame))
    report["decoded_encounters"] = int(frame["encounter_id"].nunique())
    report["rejections"] = rejection_counts
    report["manifest_counts"] = {
        f"{group}:{split_name}": int(value)
        for (group, split_name), value in frame.groupby(
            ["group", "split"], observed=True
        ).size().items()
    }
    atomic_write_json(report_out, report)
    return frame
