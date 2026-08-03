"""Leakage-safe preprocessing orchestration in the paper's fixed operation order."""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from echosynth.data.manifest import read_manifest, write_manifest
from echosynth.data.masking import sector_mask
from echosynth.data.registration import (
    CircleFit,
    SectorNotFoundError,
    align_to_reference,
    build_reference,
    detect_sector,
    fit_circular_model,
    load_reference,
)
from echosynth.utils.io import atomic_write_bytes
from echosynth.utils.logging import RejectionCounter, get_logger

LOGGER = get_logger(__name__)


def _resize_and_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    scale = size / min(height, width)
    resized_width = max(size, round(width * scale))
    resized_height = max(size, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    start_x = (resized_width - size) // 2
    start_y = (resized_height - size) // 2
    return cast(np.ndarray, resized[start_y : start_y + size, start_x : start_x + size])


def preprocess_image(
    img: np.ndarray, reference: CircleFit | None = None, image_size: int = 112
) -> np.ndarray:
    """Detect, fit, align, mask, resize, then center-crop to single-channel uint8."""
    array = np.asarray(img)
    if array.ndim != 2:
        raise ValueError(f"preprocessing expects grayscale 2-D input, got {array.shape}")
    sector = detect_sector(array)
    if sector is None:
        raise SectorNotFoundError("SECTOR_NOT_FOUND")
    fit = fit_circular_model(sector)
    target = reference or fit
    aligned = align_to_reference(array, fit, target)
    masked = sector_mask(aligned, target)
    output = _resize_and_center_crop(masked, image_size)
    return np.clip(output, 0, 255).astype(np.uint8)


def _save_png(array: np.ndarray, destination: Path) -> None:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, "PNG", optimize=False)
    atomic_write_bytes(destination, buffer.getvalue())


def _qa_sheet(images: list[np.ndarray], destination: Path) -> None:
    selected = images[:16]
    if not selected:
        return
    tile = 112
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("L", (columns * tile, rows * tile), 0)
    draw = ImageDraw.Draw(sheet)
    for index, array in enumerate(selected):
        resized = Image.fromarray(array).resize((tile, tile))
        x = (index % columns) * tile
        y = (index // columns) * tile
        sheet.paste(resized, (x, y))
        draw.text((x + 2, y + 2), str(index), fill=255)
    buffer = io.BytesIO()
    sheet.save(buffer, "PNG")
    atomic_write_bytes(destination, buffer.getvalue())


def preprocess_manifest(
    manifest: Path,
    out_dir: Path,
    reference_path: Path,
    *,
    build_reference_first: bool = False,
    qa_path: Path = Path("results/preprocess_qa.png"),
    image_size: int = 112,
) -> pd.DataFrame:
    """Preprocess every retained row and update paths with explicit failure accounting."""
    reference = (
        build_reference(manifest, reference_path)
        if build_reference_first
        else load_reference(reference_path)
    )
    frame = read_manifest(manifest)
    counter = RejectionCounter()
    outputs: list[np.ndarray] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in frame.iterrows():
        if bool(row["qc_excluded"]):
            continue
        try:
            with Image.open(str(row["png_path"])) as image:
                output = preprocess_image(np.asarray(image.convert("L")), reference, image_size)
        except SectorNotFoundError:
            counter.add("SECTOR_NOT_FOUND")
            frame.at[index, "qc_excluded"] = True
            frame.at[index, "qc_reason"] = "SECTOR_NOT_FOUND"
            LOGGER.warning("%s | SECTOR_NOT_FOUND", row["image_id"])
            continue
        destination = out_dir / f"{row['image_id']}.png"
        _save_png(output, destination)
        frame.at[index, "png_path"] = str(destination)
        outputs.append(output)
    counter.log_summary(LOGGER)
    write_manifest(frame, manifest)
    _qa_sheet(outputs, qa_path)
    return frame
