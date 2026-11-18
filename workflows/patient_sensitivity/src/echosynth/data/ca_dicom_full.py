"""PHI-safe full-cohort DICOM preparation for the AML/CA revision analysis."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixels import pixel_array as decode_pixel_array

from echosynth.data.manifest import write_manifest
from echosynth.utils.io import atomic_write_bytes, atomic_write_text


@dataclass(frozen=True)
class PrepareResult:
    """Paths and counts from one complete DICOM preparation run."""

    manifest: Path
    train_manifest: Path
    manifest_cap5: Path
    manifest_224: Path
    metadata_report: Path
    accepted_images: int
    accepted_patients: int
    accepted_encounters: int


def _hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:24]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _view_code(dataset: pydicom.dataset.Dataset) -> str:
    sequence = getattr(dataset, "ViewCodeSequence", None)
    if not sequence:
        return ""
    return _text(getattr(sequence[0], "CodeMeaning", ""))


def _ultrasound_color_data_present(dataset: pydicom.dataset.Dataset) -> bool:
    """Return the standard DICOM ultrasound color-flow indicator.

    The manuscript prespecifies metadata-based exclusion. RGB/YBR storage and small
    chromatic overlays are therefore audit variables, not exclusion criteria.
    """
    value = getattr(dataset, "UltrasoundColorDataPresent", None)
    if value is None:
        return False
    normalized = _text(value).upper()
    if normalized in {"Y", "YES", "TRUE"}:
        return True
    try:
        return int(float(normalized)) != 0
    except (TypeError, ValueError):
        return False


def _png_bytes(frame: np.ndarray, size: int) -> bytes:
    image = Image.fromarray(frame, mode="L").resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _qa_thumbnail(
    frame: np.ndarray,
    out_root: Path,
    category: str,
    group: str,
    source_hash: str,
) -> str:
    """Save a deterministic sparse QA sample without exposing a source filename."""
    if int(source_hash[:8], 16) % 4096 != 0:
        return ""
    destination = out_root / "color_qa" / category / group / f"{source_hash}.png"
    atomic_write_bytes(destination, _png_bytes(frame, 112))
    return str(destination)


def _qa_sheet(paths: list[Path], destination: Path) -> None:
    selected = paths[:64]
    if not selected:
        return
    columns = 8
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("L", (columns * 112, rows * 112), color=0)
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            tile = image.convert("L").resize((112, 112), Image.Resampling.BILINEAR)
        canvas.paste(tile, ((index % columns) * 112, (index // columns) * 112))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    atomic_write_bytes(destination, buffer.getvalue())


def _decode_grayscale_frame(
    dataset: pydicom.dataset.Dataset,
    pixels: np.ndarray,
    *,
    channel_difference_threshold: float,
) -> tuple[np.ndarray, float, bool]:
    """Return first grayscale frame plus measured chromatic-pixel fraction.

    Duke B-mode echo cine loops are frequently stored in an RGB container even when all
    three channels contain the same gray value. Container metadata alone therefore cannot
    distinguish B-mode from color Doppler. A pixel is considered chromatic only when its
    maximum channel difference exceeds the prespecified intensity threshold.
    """
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    stored_as_color = samples > 1 or str(
        getattr(dataset, "PhotometricInterpretation", "")
    ).upper().startswith(("RGB", "YBR", "PALETTE"))
    frame = np.asarray(pixels).squeeze()
    color_fraction = 0.0
    if frame.ndim == 3:
        if frame.shape[-1] in (3, 4):
            channels = frame[..., :3].astype(np.float32)
        elif frame.shape[0] in (3, 4):
            channels = np.moveaxis(frame[:3], 0, -1).astype(np.float32)
        else:
            raise ValueError(f"UNSUPPORTED_COLOR_SHAPE:{frame.shape}")
        channel_range = channels.max(axis=-1) - channels.min(axis=-1)
        color_fraction = float((channel_range > channel_difference_threshold).mean())
        frame = (
            0.299 * channels[..., 0]
            + 0.587 * channels[..., 1]
            + 0.114 * channels[..., 2]
        )
        stored_as_color = True
    elif frame.ndim != 2:
        raise ValueError(f"UNSUPPORTED_PIXEL_SHAPE:{pixels.shape}")
    values = np.nan_to_num(
        np.asarray(frame, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    nonzero = values[values != 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = float(values.min()), float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        output = np.zeros(values.shape, dtype=np.uint8)
    else:
        output = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        output = 255 - output
    return output, color_fraction, stored_as_color


def _process_one(
    payload: tuple[str, str, str, str, tuple[str, ...], float]
) -> dict[str, Any]:
    (
        group,
        path_value,
        out_root_value,
        salt,
        allowed_uids,
        channel_difference_threshold,
    ) = payload
    path = Path(path_value)
    source_hash = _hash(str(path), salt)
    try:
        header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        patient_raw = _text(getattr(header, "PatientID", ""))
        if not patient_raw:
            return {
                "accepted": False,
                "group": group,
                "source_path_hash": source_hash,
                "reason": "MISSING_PATIENT_ID",
            }
        sop_class = _text(getattr(header, "SOPClassUID", ""))
        if allowed_uids and sop_class not in allowed_uids:
            return {
                "accepted": False,
                "group": group,
                "source_path_hash": source_hash,
                "reason": "NONSTANDARD_SOP",
            }
        if _ultrasound_color_data_present(header):
            return {
                "accepted": False,
                "group": group,
                "source_path_hash": source_hash,
                "reason": "COLOR_DOPPLER_METADATA",
            }
        out_root = Path(out_root_value)
        sop_instance = _text(getattr(header, "SOPInstanceUID", "")) or source_hash
        study_uid = _text(getattr(header, "StudyInstanceUID", ""))
        study_date = _text(getattr(header, "StudyDate", ""))
        encounter_source = study_uid or f"{patient_raw}|{study_date}|{path.parent.name}"
        patient_id = "patient-" + _hash(patient_raw, salt)
        encounter_id = "encounter-" + _hash(encounter_source, salt)
        image_id = "image-" + _hash(sop_instance, salt)
        png_112 = out_root / "png_112" / group / f"{image_id}.png"
        png_224 = out_root / "png_224" / group / f"{image_id}.png"
        resume_reused = png_112.is_file() and png_224.is_file()
        color_fraction = float("nan")
        stored_as_color = int(getattr(header, "SamplesPerPixel", 1) or 1) > 1
        qa_path = ""
        if not resume_reused:
            number_of_frames = int(getattr(header, "NumberOfFrames", 1) or 1)
            # Passing the path and a frame index is essential here. It prevents
            # pydicom from materializing an entire multi-frame cine loop in memory.
            pixels = decode_pixel_array(
                path,
                index=0 if number_of_frames > 1 else None,
            )
            frame, color_fraction, stored_as_color = _decode_grayscale_frame(
                header,
                pixels,
                channel_difference_threshold=channel_difference_threshold,
            )
        else:
            frame = None
        if not resume_reused:
            if frame is None:
                raise RuntimeError("decoded frame unexpectedly absent")
            atomic_write_bytes(png_112, _png_bytes(frame, 112))
            atomic_write_bytes(png_224, _png_bytes(frame, 224))
            if stored_as_color:
                qa_path = _qa_thumbnail(
                    frame, out_root, "accepted_rgb", group, source_hash
                )
        year = int(study_date[:4]) if len(study_date) >= 4 and study_date[:4].isdigit() else None
        return {
            "accepted": True,
            "image_id": image_id,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "group": group,
            "study_date": study_date or None,
            "dicom_path": "",
            "png_path": str(png_112),
            "png_path_224": str(png_224),
            "sop_class_uid": sop_class,
            "color_doppler": False,
            "ultrasound_color_data_present": False,
            "stored_as_color": stored_as_color,
            "color_pixel_fraction": color_fraction,
            "view_pred": None,
            "view_conf": None,
            "qc_flag": False,
            "qc_features": json.dumps({}),
            "qc_excluded": False,
            "qc_reason": "",
            "split": None,
            "acquisition_year": year,
            "manufacturer": _text(getattr(header, "Manufacturer", "")),
            "manufacturer_model": _text(getattr(header, "ManufacturerModelName", "")),
            "series_description": _text(getattr(header, "SeriesDescription", "")),
            "protocol_name": _text(getattr(header, "ProtocolName", "")),
            "view_code": _view_code(header),
            "cardiac_phase": "",
            "number_of_frames": int(getattr(header, "NumberOfFrames", 1) or 1),
            "patient_id_source": "DICOM (0010,0020)",
            "source_path_hash": source_hash,
            "qa_png_path": qa_path,
            "resume_reused": resume_reused,
        }
    except Exception as error:
        return {
            "accepted": False,
            "group": group,
            "source_path_hash": source_hash,
            "reason": f"{type(error).__name__}:{error}",
        }


def _discover(
    groups: dict[str, str], extensions: set[str]
) -> tuple[list[tuple[str, Path]], int]:
    discovered: list[tuple[str, Path]] = []
    temporary_downloads_ignored = 0
    for group, root_value in groups.items():
        root = Path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"required DICOM cohort directory is missing: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            # Azure download placeholders are incomplete files, not source studies.
            # They produced the repeated FFFE/E0DD warnings and worker crashes in v4.
            if path.name.startswith(".azDownload-"):
                temporary_downloads_ignored += 1
                continue
            discovered.append((group, path))
    if not discovered:
        raise RuntimeError("no DICOM files were discovered")
    return discovered, temporary_downloads_ignored


def _worker_crash_row(payload: tuple[Any, ...]) -> dict[str, Any]:
    group, path_value, _, salt, *_ = payload
    return {
        "accepted": False,
        "group": str(group),
        "source_path_hash": _hash(str(path_value), str(salt)),
        "reason": "WORKER_CRASH_ISOLATED",
    }


def _run_process_batch(
    batch: list[tuple[Any, ...]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Decode one batch and recursively isolate any native decoder crash."""
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_process_one, batch, chunksize=1))
    except BrokenProcessPool:
        if len(batch) == 1:
            return [_worker_crash_row(batch[0])]
        midpoint = len(batch) // 2
        # Re-running a failed batch is safe because PNG writes are atomic and
        # completed PNG pairs are detected and reused without decoding.
        return _run_process_batch(batch[:midpoint], workers=workers) + _run_process_batch(
            batch[midpoint:], workers=workers
        )


def _checkpoint_matches(
    rows: list[dict[str, Any]],
    batch: list[tuple[Any, ...]],
) -> bool:
    if len(rows) != len(batch):
        return False
    for row, payload in zip(rows, batch):
        group, path_value, _, salt, *_ = payload
        if str(row.get("group")) != str(group):
            return False
        if str(row.get("source_path_hash")) != _hash(str(path_value), str(salt)):
            return False
    return True


def _bounded_process(
    payloads: Any,
    *,
    out_root: Path,
    workers: int,
    worker_batch_size: int,
) -> Any:
    """Yield crash-isolated batches with durable PHI-safe checkpoints."""
    iterator = iter(payloads)
    checkpoint_root = out_root / "prep_checkpoints_v6"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    start = 1
    while True:
        batch = list(itertools.islice(iterator, worker_batch_size))
        if not batch:
            return
        end = start + len(batch) - 1
        checkpoint = checkpoint_root / f"batch_{start:07d}_{end:07d}.jsonl"
        rows: list[dict[str, Any]] = []
        if checkpoint.is_file():
            try:
                rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line]
            except (OSError, ValueError, TypeError):
                rows = []
        checkpoint_reused = _checkpoint_matches(rows, batch)
        if not checkpoint_reused:
            rows = _run_process_batch(batch, workers=workers)
            atomic_write_text(
                checkpoint,
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            )
        for row in rows:
            row["_checkpoint_reused"] = checkpoint_reused
            if checkpoint_reused and row.get("accepted"):
                row["resume_reused"] = True
        yield from rows
        start = end + 1


def _assign_patient_splits(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    patient_groups = frame[["patient_id", "group"]].drop_duplicates()
    conflicts = patient_groups.groupby("patient_id")["group"].nunique()
    bad_patients = set(conflicts[conflicts > 1].index.astype(str))
    if bad_patients:
        frame = frame.loc[~frame["patient_id"].astype(str).isin(bad_patients)].copy()
        patient_groups = frame[["patient_id", "group"]].drop_duplicates()
    assignment: dict[str, str] = {}
    for group, patients in patient_groups.groupby("group", sort=True):
        identifiers = sorted(
            patients["patient_id"].astype(str),
            key=lambda value: hashlib.sha256(f"{seed}|{group}|{value}".encode()).hexdigest(),
        )
        total = len(identifiers)
        if total < 3:
            raise RuntimeError(f"group {group} has fewer than three unique patients")
        n_train = max(1, int(round(total * train_fraction)))
        n_validation = max(1, int(round(total * validation_fraction)))
        if n_train + n_validation >= total:
            n_train = total - 2
            n_validation = 1
        for patient_id in identifiers[:n_train]:
            assignment[patient_id] = "train"
        for patient_id in identifiers[n_train : n_train + n_validation]:
            assignment[patient_id] = "val"
        for patient_id in identifiers[n_train + n_validation :]:
            assignment[patient_id] = "test"
    output = frame.copy()
    output["split"] = output["patient_id"].astype(str).map(assignment)
    if output["split"].isna().any():
        raise RuntimeError("split assignment failed for one or more accepted images")
    leakage = output.groupby("patient_id")["split"].nunique()
    if (leakage > 1).any():
        raise RuntimeError("patient leakage detected after patient-level splitting")
    output.attrs["conflicting_patients_excluded"] = len(bad_patients)
    return output


def _completeness(frame: pd.DataFrame, column: str) -> float:
    values = frame[column]
    return float(values.notna().mul(values.astype(str).str.strip().ne("")).mean())


def prepare_full_cohort(config: dict[str, Any], out_root: Path) -> PrepareResult:
    """Decode the full source archive, hash identifiers, split patients, and write manifests."""
    data = dict(config["data"])
    groups = {str(key): str(value) for key, value in data["groups"].items()}
    expected = {"CA", "LVH", "GEN"}
    if set(groups) != expected:
        raise ValueError(f"groups must be exactly {sorted(expected)}")
    extensions = {str(value).lower() for value in data.get("extensions", [".dcm"])}
    salt = str(config["privacy"]["hash_salt"])
    if len(salt) < 16 or salt.lower().startswith("change"):
        raise ValueError("privacy.hash_salt must be a fixed non-placeholder string of 16+ characters")
    allowed_uids = tuple(str(value) for value in data.get("standard_sop_class_uids", []))
    channel_difference_threshold = float(data.get("channel_difference_threshold", 12.0))
    if channel_difference_threshold < 0:
        raise ValueError("invalid pixel-color audit threshold")
    discovered, temporary_downloads_ignored = _discover(groups, extensions)
    payloads = (
        (
            group,
            str(path),
            str(out_root),
            salt,
            allowed_uids,
            channel_difference_threshold,
        )
        for group, path in discovered
    )
    workers = int(config.get("workers", 2))
    result_iterator = _bounded_process(
        payloads,
        out_root=out_root,
        workers=max(1, workers),
        worker_batch_size=int(config.get("worker_batch_size", 1000)),
    )
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    checkpoint_rows_reused = 0
    for index, row in enumerate(result_iterator, start=1):
        checkpoint_rows_reused += int(bool(row.pop("_checkpoint_reused", False)))
        (accepted_rows if row.get("accepted") else rejected_rows).append(row)
        if index % 10000 == 0:
            print(
                f"DICOM PREP PROGRESS processed={index}/{len(discovered)} "
                f"accepted={len(accepted_rows)} rejected={len(rejected_rows)}",
                flush=True,
            )
    accepted = pd.DataFrame(accepted_rows).drop(columns="accepted")
    rejected = pd.DataFrame(rejected_rows).drop(columns="accepted", errors="ignore")
    del accepted_rows, rejected_rows
    if accepted.empty:
        raise RuntimeError("all discovered DICOM files were rejected")
    duplicate_mask = accepted.duplicated("image_id", keep="first")
    duplicate_rows = accepted.loc[duplicate_mask, ["group", "source_path_hash"]].copy()
    if not duplicate_rows.empty:
        duplicate_rows["reason"] = "DUPLICATE_SOP_INSTANCE"
        rejected = pd.concat([rejected, duplicate_rows], ignore_index=True)
    accepted = accepted.loc[~duplicate_mask].copy()
    accepted = _assign_patient_splits(
        accepted,
        seed=int(config.get("seed", 2024)),
        train_fraction=float(data.get("train_fraction", 0.70)),
        validation_fraction=float(data.get("validation_fraction", 0.15)),
    )
    accepted = accepted.sort_values(["split", "group", "patient_id", "encounter_id", "image_id"])
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    full_path = manifest_dir / "ca_master.csv"
    write_manifest(accepted, full_path)
    train_path = manifest_dir / "ca_train.csv"
    write_manifest(accepted.loc[accepted["split"].astype(str) == "train"].copy(), train_path)
    cap5 = accepted.groupby("encounter_id", sort=False, group_keys=False).head(5).copy()
    cap5_path = manifest_dir / "ca_master_cap5.csv"
    write_manifest(cap5, cap5_path)
    high_res = accepted.copy()
    high_res["png_path"] = high_res["png_path_224"]
    high_res_path = manifest_dir / "ca_master_224.csv"
    write_manifest(high_res, high_res_path)
    rejected.to_csv(out_root / "decode_rejections.csv", index=False)
    accepted_qa = [
        Path(value)
        for value in accepted.get("qa_png_path", pd.Series(dtype=str)).dropna().astype(str)
        if value
    ]
    rejected_qa = [
        Path(value)
        for value in rejected.get("qa_png_path", pd.Series(dtype=str)).dropna().astype(str)
        if value
    ]
    _qa_sheet(accepted_qa, out_root / "color_qa_accepted_rgb_v6.png")
    _qa_sheet(rejected_qa, out_root / "color_qa_rejected_doppler_v6.png")
    patients = accepted[["patient_id", "group", "split"]].drop_duplicates()
    encounters = accepted[["encounter_id", "patient_id", "group", "split"]].drop_duplicates()
    counts = (
        encounters.groupby(["group", "split"]).size().rename("n_encounters").reset_index()
    )
    counts.to_csv(out_root / "cohort_counts.csv", index=False)
    source_counts = pd.Series([group for group, _ in discovered]).value_counts().to_dict()
    accepted_counts = accepted["group"].astype(str).value_counts().to_dict()
    rejection_counts = (
        rejected["reason"].astype(str).value_counts().to_dict() if not rejected.empty else {}
    )
    rejection_by_group = (
        rejected.groupby(["group", "reason"], dropna=False).size().to_dict()
        if not rejected.empty
        else {}
    )
    report = {
        "source_files_discovered": len(discovered),
        "temporary_download_files_ignored": temporary_downloads_ignored,
        "checkpoint_rows_reused": checkpoint_rows_reused,
        "accepted_images": len(accepted),
        "rejected_files": len(rejected),
        "source_files_by_group": {str(key): int(value) for key, value in source_counts.items()},
        "accepted_images_by_group": {
            str(key): int(value) for key, value in accepted_counts.items()
        },
        "rejection_counts": {str(key): int(value) for key, value in rejection_counts.items()},
        "rejection_counts_by_group": {
            f"{group}|{reason}": int(value)
            for (group, reason), value in rejection_by_group.items()
        },
        "rgb_container_images_accepted_under_metadata_rule": int(
            accepted["stored_as_color"].fillna(False).astype(bool).sum()
        ),
        "resumed_images_reused": int(
            accepted["resume_reused"].fillna(False).astype(bool).sum()
        ),
        "color_qa": {
            "accepted_rgb_thumbnails": len(accepted_qa),
            "rejected_color_thumbnails": len(rejected_qa),
            "accepted_contact_sheet": str(out_root / "color_qa_accepted_rgb_v6.png"),
            "rejected_contact_sheet": str(out_root / "color_qa_rejected_doppler_v6.png"),
            "sampling_rule": "SHA-256 source hash modulo 4096 equals zero; maximum 64 per sheet",
        },
        "color_exclusion_rule": {
            "dicom_tag": "UltrasoundColorDataPresent (0028,0014)",
            "definition": (
                "Exclude only when the standard DICOM ultrasound color-data indicator "
                "is present and nonzero, matching the manuscript Methods."
            ),
        },
        "chromatic_pixel_audit": {
            "channel_difference_threshold": channel_difference_threshold,
            "definition": (
                "Record the fraction of pixels whose maximum RGB channel difference exceeds "
                "the channel threshold; this value is descriptive and does not exclude images."
            ),
        },
        "accepted_patients": int(patients["patient_id"].nunique()),
        "accepted_encounters": int(encounters["encounter_id"].nunique()),
        "patient_leakage_count": int((accepted.groupby("patient_id")["split"].nunique() > 1).sum()),
        "conflicting_label_patients_excluded": int(
            accepted.attrs.get("conflicting_patients_excluded", 0)
        ),
        "patient_id_source": "DICOM (0010,0020), SHA-256 hashed before output",
        "acquisition_year_completeness": _completeness(accepted, "acquisition_year"),
        "manufacturer_completeness": _completeness(accepted, "manufacturer"),
        "manufacturer_model_completeness": _completeness(accepted, "manufacturer_model"),
        "standardized_view_completeness": _completeness(accepted, "view_code"),
        "standardized_cardiac_phase_completeness": _completeness(accepted, "cardiac_phase"),
        "split_policy": {
            "unit": "unique DICOM PatientID",
            "train_fraction": float(data.get("train_fraction", 0.70)),
            "validation_fraction": float(data.get("validation_fraction", 0.15)),
            "test_fraction": 1.0
            - float(data.get("train_fraction", 0.70))
            - float(data.get("validation_fraction", 0.15)),
            "seed": int(config.get("seed", 2024)),
            "stratified_by": "CA/LVH/GEN cohort label",
        },
    }
    minimum_encounters = {
        str(key): int(value) for key, value in data.get("minimum_encounters", {}).items()
    }
    actual_encounters = encounters["group"].astype(str).value_counts().to_dict()
    deficiencies = {
        group: {"actual": int(actual_encounters.get(group, 0)), "minimum": minimum}
        for group, minimum in minimum_encounters.items()
        if int(actual_encounters.get(group, 0)) < minimum
    }
    report["cohort_size_gate"] = {
        "status": "PASS" if not deficiencies else "FAIL",
        "minimum_encounters": minimum_encounters,
        "actual_encounters": {
            str(key): int(value) for key, value in actual_encounters.items()
        },
        "deficiencies": deficiencies,
    }
    report_path = out_root / "metadata_completeness.json"
    atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    if deficiencies:
        raise RuntimeError(
            "cohort-size safety gate failed; inspect metadata_completeness.json: "
            f"{deficiencies}"
        )
    return PrepareResult(
        full_path,
        train_path,
        cap5_path,
        high_res_path,
        report_path,
        len(accepted),
        int(patients["patient_id"].nunique()),
        int(encounters["encounter_id"].nunique()),
    )
