"""DICOM discovery, validation, decoding, and manifest generation (paper section 2.3)."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pydicom
import yaml
from PIL import Image
from pydicom.dataset import Dataset

from echosynth.data.manifest import ManifestSchema, write_manifest
from echosynth.utils.io import atomic_write_bytes
from echosynth.utils.logging import RejectionCounter, get_logger

LOGGER = get_logger(__name__)
REASON_CODES = (
    "UNREADABLE",
    "NONSTANDARD_SOP",
    "COLOR_DOPPLER",
    "NO_PIXEL_DATA",
    "DECODE_FAILED",
    "DUPLICATE_ID",
)


@dataclass(frozen=True)
class _Decoded:
    path: Path
    row: dict[str, Any] | None
    pixels: np.ndarray | None
    reason: str | None


def scan_archive(root: Path) -> pd.DataFrame:
    """Discover source files recursively without loading pixel data."""
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return pd.DataFrame({"dicom_path": [str(path) for path in paths]})


def is_standard_acquisition(ds: Dataset, allowed_uids: Collection[str] | None = None) -> bool:
    """Check SOP Class UID against a config-supplied allowlist; none is encoded in code."""
    if allowed_uids is None:
        raise ValueError("allowed_uids must come from the data configuration")
    return str(ds.get("SOPClassUID", "")) in set(allowed_uids)


def has_color_doppler(ds: Dataset) -> bool:
    """Detect color using the standard flag, multichannel photometric mode, or palette hints.

    The heuristics are, in order: ``UltrasoundColorDataPresent``; more than one sample per
    pixel; RGB/YBR/PALETTE photometric interpretations; and any palette-color lookup-table
    descriptor. These metadata checks avoid diagnostic interpretation of pixels.
    """
    if int(ds.get("UltrasoundColorDataPresent", 0) or 0) == 1:
        return True
    if int(ds.get("SamplesPerPixel", 1) or 1) > 1:
        return True
    photometric = str(ds.get("PhotometricInterpretation", "")).upper()
    if photometric.startswith(("RGB", "YBR", "PALETTE")):
        return True
    return any(
        tag in ds
        for tag in (
            (0x0028, 0x1101),
            (0x0028, 0x1102),
            (0x0028, 0x1103),
            (0x0028, 0x1201),
            (0x0028, 0x1202),
            (0x0028, 0x1203),
        )
    )


def extract_first_frame(ds: Dataset) -> np.ndarray:
    """Extract frame index zero by predefined rule, not diagnostic-phase optimization.

    Implements paper line 145. Pydicom dispatches compressed transfer syntaxes through its
    installed pylibjpeg/GDCM handlers; decoding errors are surfaced to the caller.
    """
    if "PixelData" not in ds:
        raise ValueError("NO_PIXEL_DATA")
    pixels = np.asarray(ds.pixel_array)
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    frame = pixels[0] if frames > 1 else pixels
    if frame.ndim != 2:
        raise ValueError(f"decoded frame must be grayscale 2-D, got {frame.shape}")
    values = frame.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("decoded frame has no finite pixels")
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high > low:
        values = (values - low) * (255.0 / (high - low))
    else:
        values = np.zeros_like(values)
    values[~finite] = 0
    output = np.clip(values, 0, 255).astype(np.uint8)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        output = 255 - output
    return cast(np.ndarray, output)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _decode(path: Path, allowed_uids: Collection[str]) -> _Decoded:
    try:
        ds = pydicom.dcmread(path, force=False)
    except Exception as error:  # pydicom exposes several parser-specific exception classes
        LOGGER.warning("%s | UNREADABLE | %s", path, error)
        return _Decoded(path, None, None, "UNREADABLE")
    if not is_standard_acquisition(ds, allowed_uids):
        return _Decoded(path, None, None, "NONSTANDARD_SOP")
    if has_color_doppler(ds):
        return _Decoded(path, None, None, "COLOR_DOPPLER")
    if "PixelData" not in ds:
        return _Decoded(path, None, None, "NO_PIXEL_DATA")
    try:
        pixels = extract_first_frame(ds)
    except Exception as error:
        LOGGER.warning("%s | DECODE_FAILED | %s", path, error)
        return _Decoded(path, None, None, "DECODE_FAILED")
    sop_instance = str(ds.get("SOPInstanceUID", path.resolve()))
    image_id = _stable_hash(sop_instance)
    row: dict[str, Any] = {
        "image_id": image_id,
        "patient_id": _stable_hash(str(ds.get("PatientID", "missing-patient"))),
        "encounter_id": _stable_hash(str(ds.get("StudyInstanceUID", sop_instance))),
        "group": str(ds.get("ClinicalTrialSubjectReadingID", "GEN")),
        "study_date": str(ds.get("StudyDate", "")) or None,
        "dicom_path": str(path),
        "png_path": "",
        "sop_class_uid": str(ds.get("SOPClassUID", "")),
        "color_doppler": False,
        "view_pred": None,
        "view_conf": None,
        "qc_flag": False,
        "qc_features": json.dumps({}),
        "qc_excluded": False,
        "qc_reason": "",
        "split": None,
    }
    if row["group"] not in ManifestSchema.groups:
        row["group"] = "GEN"
    return _Decoded(path, row, pixels, None)


def _png_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def ingest(
    root: Path,
    out_dir: Path,
    manifest_out: Path,
    n_workers: int = 4,
    *,
    config_path: Path = Path("configs/data/duke.yaml"),
    dry_run: bool = False,
) -> pd.DataFrame:
    """Decode files in parallel, log every rejection, and write a validated manifest."""
    if n_workers < 1:
        raise ValueError("n_workers must be at least one")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed_uids = config.get("standard_sop_class_uids", [])
    if not allowed_uids:
        raise ValueError(f"No standard_sop_class_uids configured in {config_path}")
    paths = [Path(value) for value in scan_archive(root)["dicom_path"].tolist()]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        decoded = list(executor.map(lambda path: _decode(path, allowed_uids), paths))
    counter = RejectionCounter()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in decoded:
        if item.reason is not None:
            counter.add(item.reason)
            LOGGER.warning("%s | %s", item.path, item.reason)
            continue
        if item.row is None or item.pixels is None:
            raise RuntimeError(f"decoder returned incomplete success for {item.path}")
        image_id = str(item.row["image_id"])
        if image_id in seen:
            counter.add("DUPLICATE_ID")
            LOGGER.warning("%s | DUPLICATE_ID | %s", item.path, image_id)
            continue
        seen.add(image_id)
        destination = out_dir / f"{image_id}.png"
        item.row["png_path"] = str(destination)
        rows.append(item.row)
        if not dry_run:
            atomic_write_bytes(destination, _png_bytes(item.pixels))
    frame = pd.DataFrame(rows, columns=ManifestSchema.columns)
    frame.attrs["rejections"] = dict(counter)
    counter.log_summary(LOGGER)
    if not dry_run:
        write_manifest(frame, manifest_out)
    return frame
