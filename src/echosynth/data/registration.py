"""Ultrasound-sector detection and geometric registration (paper lines 166-173)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from echosynth.data.manifest import LeakageError, read_manifest
from echosynth.utils.io import atomic_write_json


@dataclass(frozen=True)
class Sector:
    """Detected fan geometry and sampled outer-arc coordinates."""

    apex_x: float
    apex_y: float
    inner_radius: float
    outer_radius: float
    start_angle: float
    end_angle: float
    outer_arc: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class CircleFit:
    """Circular sector model used for deterministic registration and masking."""

    center_x: float
    center_y: float
    radius: float
    angle: float
    opening_angle: float
    inner_radius: float = 0.0


class SectorNotFoundError(ValueError):
    """Raised when preprocessing cannot detect an ultrasound sector."""


def _vertex_angle(previous: np.ndarray, vertex: np.ndarray, following: np.ndarray) -> float:
    left = previous - vertex
    right = following - vertex
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return math.pi
    cosine = float(np.clip(np.dot(left, right) / denominator, -1, 1))
    return float(math.acos(cosine))


def _minimal_angle_interval(angles: np.ndarray) -> tuple[float, float]:
    normalized = np.sort(np.mod(angles, 2 * math.pi))
    extended = np.r_[normalized, normalized[0] + 2 * math.pi]
    gaps = np.diff(extended)
    gap_index = int(np.argmax(gaps))
    start = float(extended[gap_index + 1] % (2 * math.pi))
    end = float(extended[gap_index] % (2 * math.pi))
    if end <= start:
        end += 2 * math.pi
    return start, end


def detect_sector(img: np.ndarray) -> Sector | None:
    """Detect the largest ultrasound wedge, returning None instead of guessing on failure."""
    array = np.asarray(img)
    if array.ndim != 2 or array.size == 0:
        return None
    threshold = max(1, int(np.percentile(array, 20)))
    support = (array > threshold).astype(np.uint8) * 255
    support = cast(
        np.ndarray,
        cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
    )
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < array.size * 0.01:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, max(1.0, 0.02 * perimeter), True).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    angles = [
        _vertex_angle(polygon[index - 1], polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    apex = polygon[int(np.argmin(angles))].astype(float)
    points = contour.reshape(-1, 2).astype(float)
    deltas = points - apex
    distances = np.linalg.norm(deltas, axis=1)
    if float(distances.max()) < 5:
        return None
    radial_angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    start, end = _minimal_angle_interval(radial_angles[distances > 2])
    opening = end - start
    if not (math.radians(10) <= opening <= math.radians(170)):
        return None
    outer_threshold = float(np.quantile(distances, 0.75))
    outer_points = points[distances >= outer_threshold]
    nonzero_y, nonzero_x = np.nonzero(support)
    all_distances = np.hypot(nonzero_x - apex[0], nonzero_y - apex[1])
    positive = all_distances[all_distances > 0]
    inner_radius = float(np.quantile(positive, 0.01)) if positive.size else 0.0
    return Sector(
        apex_x=float(apex[0]),
        apex_y=float(apex[1]),
        inner_radius=inner_radius,
        outer_radius=float(distances.max()),
        start_angle=start,
        end_angle=end,
        outer_arc=tuple((float(x), float(y)) for x, y in outer_points),
    )


def fit_circular_model(sector: Sector) -> CircleFit:
    """Fit the sampled outer arc by algebraic least squares."""
    if len(sector.outer_arc) >= 3:
        points = np.asarray(sector.outer_arc, dtype=float)
        matrix = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]
        target = points[:, 0] ** 2 + points[:, 1] ** 2
        center_x, center_y, offset = np.linalg.lstsq(matrix, target, rcond=None)[0]
        radius_sq = offset + center_x**2 + center_y**2
        radius = math.sqrt(max(float(radius_sq), 1.0))
    else:
        center_x, center_y, radius = sector.apex_x, sector.apex_y, sector.outer_radius
    angle = sector.start_angle + (sector.end_angle - sector.start_angle) / 2
    return CircleFit(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        angle=float(angle),
        opening_angle=float(sector.end_angle - sector.start_angle),
        inner_radius=sector.inner_radius,
    )


def align_to_reference(img: np.ndarray, fit: CircleFit, ref: CircleFit) -> np.ndarray:
    """Apply rotation, scale, and translation to match cached training geometry."""
    if fit.radius <= 0 or ref.radius <= 0:
        raise ValueError("circle radii must be positive")
    rotation_degrees = math.degrees(ref.angle - fit.angle)
    matrix = cv2.getRotationMatrix2D(
        (fit.center_x, fit.center_y), rotation_degrees, ref.radius / fit.radius
    )
    transformed_center = matrix @ np.array([fit.center_x, fit.center_y, 1.0])
    matrix[0, 2] += ref.center_x - transformed_center[0]
    matrix[1, 2] += ref.center_y - transformed_center[1]
    height, width = img.shape[:2]
    output = cv2.warpAffine(
        np.asarray(img),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    return cast(np.ndarray, output)


def median_reference(fits: list[CircleFit]) -> CircleFit:
    """Return the component-wise median geometry from training images only."""
    if not fits:
        raise ValueError("cannot build a reference without successful sector fits")
    values = np.array(
        [
            [fit.center_x, fit.center_y, fit.radius, fit.angle, fit.opening_angle, fit.inner_radius]
            for fit in fits
        ]
    )
    median = np.median(values, axis=0)
    return CircleFit(*map(float, median))


def build_reference(manifest: Path, cache: Path) -> CircleFit:
    """Build and cache median geometry, rejecting any non-training manifest row."""
    frame = read_manifest(manifest)
    splits = set(frame["split"].dropna().astype(str))
    if splits != {"train"}:
        raise LeakageError(f"Reference may be built from train only; found splits {sorted(splits)}")
    fits: list[CircleFit] = []
    for path in frame["png_path"].astype(str):
        with Image.open(path) as image:
            sector = detect_sector(np.asarray(image.convert("L")))
        if sector is not None:
            fits.append(fit_circular_model(sector))
    reference = median_reference(fits)
    atomic_write_json(cache, asdict(reference))
    return reference


def load_reference(cache: Path) -> CircleFit:
    """Load a previously cached training reference."""
    import json

    payload: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
    return CircleFit(**payload)
