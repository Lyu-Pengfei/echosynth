"""Validated CSV data contracts and leakage invariants (spec sections 3.1 and 3.4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from echosynth.utils.io import atomic_write_text


class ManifestError(ValueError):
    """Raised when a manifest violates its schema."""


class LeakageError(ManifestError):
    """Raised when a manifest violates a partition invariant."""


@dataclass(frozen=True)
class ManifestSchema:
    """The exact Duke master-manifest contract from spec section 3.1."""

    columns: ClassVar[tuple[str, ...]] = (
        "image_id",
        "patient_id",
        "encounter_id",
        "group",
        "study_date",
        "dicom_path",
        "png_path",
        "sop_class_uid",
        "color_doppler",
        "view_pred",
        "view_conf",
        "qc_flag",
        "qc_features",
        "qc_excluded",
        "qc_reason",
        "split",
    )
    groups: ClassVar[frozenset[str]] = frozenset({"CA", "HCM", "LVH", "GEN"})
    views: ClassVar[frozenset[str]] = frozenset(
        {"plax", "psax-av", "psax-mv", "psax-ap", "a4c", "a5c", "a3c", "a2c"}
    )
    splits: ClassVar[frozenset[str]] = frozenset({"train", "val", "test"})


def _bad_values(series: pd.Series, allowed: frozenset[str]) -> list[str]:
    values = series.dropna().astype(str)
    return sorted(set(values) - allowed)


def validate_manifest(df: pd.DataFrame, *, require_all_columns: bool = True) -> pd.DataFrame:
    """Validate schema types/enums and return a normalized copy."""
    missing = [column for column in ManifestSchema.columns if column not in df.columns]
    if require_all_columns and missing:
        raise ManifestError(f"Missing manifest columns: {missing}")
    result = df.copy()
    required_ids = ["image_id", "patient_id", "encounter_id"]
    present_ids = [column for column in required_ids if column in result]
    for column in present_ids:
        if result[column].isna().any() or (result[column].astype(str).str.len() == 0).any():
            raise ManifestError(f"{column} contains null or empty values")
        result[column] = result[column].astype("string")
    if "image_id" in result and result["image_id"].duplicated().any():
        duplicates = sorted(result.loc[result["image_id"].duplicated(False), "image_id"].unique())
        raise ManifestError(f"image_id is not unique: {duplicates[:20]}")
    for column in ("dicom_path", "png_path", "sop_class_uid", "qc_reason"):
        if column in result:
            result[column] = result[column].astype("string")
    if "group" in result:
        bad = _bad_values(result["group"], ManifestSchema.groups)
        if bad:
            raise ManifestError(f"Invalid group values: {bad}")
        result["group"] = pd.Categorical(result["group"], categories=sorted(ManifestSchema.groups))
    if "view_pred" in result:
        bad = _bad_values(result["view_pred"], ManifestSchema.views)
        if bad:
            raise ManifestError(f"Invalid view_pred values: {bad}")
        result["view_pred"] = pd.Categorical(
            result["view_pred"], categories=sorted(ManifestSchema.views)
        )
    if "split" in result:
        bad = _bad_values(result["split"], ManifestSchema.splits)
        if bad:
            raise ManifestError(f"Invalid split values: {bad}")
        result["split"] = pd.Categorical(result["split"], categories=["train", "val", "test"])
    if "study_date" in result:
        converted = pd.to_datetime(result["study_date"], errors="coerce")
        invalid_dates = converted.isna() & result["study_date"].notna()
        if invalid_dates.any():
            bad_ids = result.loc[invalid_dates, "image_id"].astype(str).tolist()
            raise ManifestError(f"Invalid study_date for image_ids: {bad_ids[:20]}")
        result["study_date"] = converted.dt.date
    for column in ("color_doppler", "qc_flag", "qc_excluded"):
        if column in result:
            invalid = result[column].dropna().map(lambda value: not isinstance(value, bool | int))
            if invalid.any():
                raise ManifestError(f"{column} must contain booleans")
            result[column] = result[column].astype("boolean")
    if "view_conf" in result:
        result["view_conf"] = pd.to_numeric(result["view_conf"], errors="coerce").astype("Float64")
        nonnull = result["view_conf"].dropna()
        if ((nonnull < 0) | (nonnull > 1)).any():
            raise ManifestError("view_conf must be in [0, 1]")
    if "qc_features" in result:
        for image_id, payload in zip(
            result.get("image_id", result.index), result["qc_features"], strict=True
        ):
            if pd.isna(payload) or isinstance(payload, dict):
                continue
            try:
                decoded = json.loads(str(payload))
            except json.JSONDecodeError as error:
                raise ManifestError(f"Invalid qc_features JSON for image_id {image_id}") from error
            if not isinstance(decoded, dict):
                raise ManifestError(f"qc_features must be an object for image_id {image_id}")
    return result


def read_manifest(path: Path | str) -> pd.DataFrame:
    """Read, validate, and leakage-check a Duke master manifest."""
    frame = pd.read_csv(path, keep_default_na=True)
    validated = validate_manifest(frame)
    assert_manifest_invariants(validated)
    return validated


def write_manifest(df: pd.DataFrame, path: Path | str) -> None:
    """Validate on write and atomically replace a Duke master manifest."""
    validated = validate_manifest(df)
    assert_manifest_invariants(validated)
    serializable = validated.copy()
    if "study_date" in serializable:
        serializable["study_date"] = serializable["study_date"].map(
            lambda value: value.isoformat() if pd.notna(value) else ""
        )
    atomic_write_text(Path(path), serializable.to_csv(index=False))


def _ids_spanning_splits(df: pd.DataFrame, column: str) -> list[str]:
    available = df.dropna(subset=["split"])
    counts = available.groupby(column, observed=True)["split"].nunique()
    return sorted(counts[counts > 1].index.astype(str).tolist())


def assert_manifest_invariants(df: pd.DataFrame) -> None:
    """Enforce the five leakage/count invariants in spec section 3.4."""
    if "split" not in df:
        return
    for column in ("encounter_id", "patient_id"):
        if column in df:
            offenders = _ids_spanning_splits(df, column)
            if offenders:
                raise LeakageError(f"{column} spans multiple splits: {offenders[:20]}")
    if "png_path" in df:
        path_splits = (
            df.dropna(subset=["png_path", "split"])
            .groupby("png_path", observed=True)["split"]
            .nunique()
        )
        offenders = sorted(path_splits[path_splits > 1].index.astype(str).tolist())
        if offenders:
            raise LeakageError(f"png_path appears in multiple splits: {offenders[:20]}")
    synthetic_mask = pd.Series(False, index=df.index)
    if "is_synthetic" in df:
        synthetic_mask |= df["is_synthetic"].fillna(False).astype(bool)
    if "generator_run_id" in df:
        synthetic_mask |= df["generator_run_id"].notna()
    if "task" in df and "dicom_path" not in df:
        synthetic_mask |= df["task"].notna()
    bad_synthetic = synthetic_mask & df["split"].isin(["val", "test"])
    if bad_synthetic.any():
        ids = df.loc[bad_synthetic, "image_id"].astype(str).tolist()
        raise LeakageError(f"Synthetic images outside train split: {ids[:20]}")
    if {"group", "encounter_id"}.issubset(df) and df["split"].notna().all():
        encounters = df[["encounter_id", "group", "split"]].drop_duplicates()
        if (
            len(encounters) == 4697
            and set(encounters["group"].astype(str)) == ManifestSchema.groups
        ):
            expected = {
                ("CA", "train"): 216,
                ("CA", "val"): 46,
                ("CA", "test"): 47,
                ("GEN", "train"): 1542,
                ("GEN", "val"): 330,
                ("GEN", "test"): 331,
                ("LVH", "train"): 1529,
                ("LVH", "val"): 328,
                ("LVH", "test"): 328,
            }
            actual = encounters.groupby(["group", "split"], observed=True).size().to_dict()
            if actual != expected:
                raise LeakageError(
                    f"Duke Table 1 split counts differ: {actual}; expected {expected}"
                )
