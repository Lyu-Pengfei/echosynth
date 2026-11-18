"""Build a leakage-safe first-frame manifest from the PRCC echo ``.npy`` archive."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from echosynth.data.manifest import assert_manifest_invariants, write_manifest


GROUP_DIRECTORIES = {
    "CA": "aml",
    "LVH": "aml_lvh_control",
    "GEN": "aml_general_control",
}


def _hash(value: str) -> str:
    return hashlib.sha256(("echosynth-prcc-v2|" + value).encode()).hexdigest()[:20]


def _first_frame(array: np.ndarray) -> np.ndarray:
    """Return one grayscale frame from common echo array layouts."""
    frame = np.asarray(array).squeeze()
    while frame.ndim > 3:
        frame = frame[0]
    if frame.ndim == 3:
        if frame.shape[-1] in (3, 4):
            frame = frame[..., :3].mean(axis=-1)
        elif frame.shape[0] in (3, 4):
            frame = frame[:3].mean(axis=0)
        else:
            temporal_axis = int(np.argmin(frame.shape))
            frame = np.take(frame, 0, axis=temporal_axis)
    if frame.ndim != 2:
        raise ValueError(f"unsupported array shape after squeeze: {np.asarray(array).shape}")
    frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = frame[frame != 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1.0, 99.0])
    else:
        low, high = float(frame.min()), float(frame.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(frame.shape, dtype=np.uint8)
    return np.clip((frame - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _patient_assignments(
    patients: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not math.isclose(
        sum(ratios), 1.0
    ):
        raise ValueError("split ratios must be three positive values summing to one")
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, group in patients.groupby("group", sort=True):
        ids = np.asarray(sorted(group["patient_id"].astype(str).unique()), dtype=object)
        ids = ids[rng.permutation(len(ids))]
        n_train = math.floor(len(ids) * ratios[0])
        n_test = math.ceil(len(ids) * ratios[2])
        n_val = len(ids) - n_train - n_test
        for patient_id in ids[:n_train]:
            assignments[str(patient_id)] = "train"
        for patient_id in ids[n_train : n_train + n_val]:
            assignments[str(patient_id)] = "val"
        for patient_id in ids[n_train + n_val :]:
            assignments[str(patient_id)] = "test"
    return assignments


def build_prcc_manifest(
    root: Path,
    out_dir: Path,
    manifest_out: Path,
    train_manifest_out: Path,
    report_out: Path,
    *,
    seed: int = 2024,
    image_size: int = 112,
    max_images_per_patient: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Scan CA/LVH/general controls and export a patient-level first-frame dataset."""
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for group, directory in GROUP_DIRECTORIES.items():
        group_root = root / directory
        if not group_root.is_dir():
            raise FileNotFoundError(f"missing PRCC group directory: {group_root}")
        by_patient: dict[str, list[Path]] = {}
        for path in sorted(group_root.rglob("*.npy")):
            relative = path.relative_to(group_root)
            raw_patient = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
            by_patient.setdefault(raw_patient, []).append(path)
        for raw_patient, paths in sorted(by_patient.items()):
            selected = paths[:max_images_per_patient] if max_images_per_patient > 0 else paths
            patient_id = "p-" + _hash(f"{group}|{raw_patient}")
            for source in selected:
                relative = source.relative_to(group_root)
                image_id = "i-" + _hash(f"{group}|{relative.as_posix()}")
                png_path = (out_dir / group.lower() / patient_id / f"{image_id}.png").resolve()
                try:
                    array = np.load(source, mmap_mode="r", allow_pickle=False)
                    frame = _first_frame(array)
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).resize(
                        (image_size, image_size), resample=Image.Resampling.BILINEAR
                    ).save(png_path)
                except Exception as error:  # keep an auditable rejection log
                    failures.append({"source": str(source), "reason": repr(error)})
                    continue
                rows.append(
                    {
                        "image_id": image_id,
                        "patient_id": patient_id,
                        # Patient is the conservative evaluation unit for the revision.
                        "encounter_id": patient_id,
                        "group": group,
                        "study_date": None,
                        "dicom_path": str(source),
                        "png_path": str(png_path),
                        "sop_class_uid": "npy-derived",
                        "color_doppler": False,
                        "view_pred": None,
                        "view_conf": None,
                        "qc_flag": False,
                        "qc_features": json.dumps({}),
                        "qc_excluded": False,
                        "qc_reason": "",
                        "split": None,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no readable .npy files found below {root}")
    conflicts = frame.groupby("patient_id")["group"].nunique()
    if (conflicts > 1).any():
        raise RuntimeError("a hashed patient appears in more than one phenotype group")
    assignments = _patient_assignments(
        frame[["patient_id", "group"]].drop_duplicates(), seed, ratios
    )
    frame["split"] = frame["patient_id"].map(assignments)
    assert_manifest_invariants(frame)
    write_manifest(frame, manifest_out)
    write_manifest(frame.loc[frame["split"] == "train"].copy(), train_manifest_out)
    summary = {
        "source_root": str(root),
        "seed": seed,
        "image_size": image_size,
        "max_images_per_patient": max_images_per_patient,
        "n_images": int(len(frame)),
        "n_patients": int(frame["patient_id"].nunique()),
        "counts_by_group_split": {
            f"{group}|{split}": int(count)
            for (group, split), count in frame.groupby(["group", "split"]).size().items()
        },
        "patients_by_group_split": {
            f"{group}|{split}": int(count)
            for (group, split), count in (
                frame[["patient_id", "group", "split"]]
                .drop_duplicates()
                .groupby(["group", "split"])
                .size()
                .items()
            )
        },
        "n_decode_failures": len(failures),
        "decode_failures_first_100": failures[:100],
        "patient_leakage_count": int(
            (frame.groupby("patient_id")["split"].nunique() > 1).sum()
        ),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(summary, indent=2) + "\n")
    return frame
