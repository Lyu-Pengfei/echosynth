"""Sector masking that removes annotations outside acquisition geometry."""

from __future__ import annotations

from typing import cast

import numpy as np

from echosynth.data.registration import CircleFit


def sector_mask(img: np.ndarray, fit: CircleFit) -> np.ndarray:
    """Zero pixels outside the fitted wedge; applying twice is idempotent."""
    array = np.asarray(img)
    height, width = array.shape[:2]
    y_coordinates, x_coordinates = np.ogrid[:height, :width]
    delta_x = x_coordinates - fit.center_x
    delta_y = y_coordinates - fit.center_y
    radius = np.hypot(delta_x, delta_y)
    angles = np.arctan2(delta_y, delta_x)
    angular_delta = np.angle(np.exp(1j * (angles - fit.angle)))
    keep = (
        (radius >= max(0, fit.inner_radius))
        & (radius <= fit.radius)
        & (np.abs(angular_delta) <= fit.opening_angle / 2)
    )
    output = np.zeros_like(array)
    output[keep] = array[keep]
    return cast(np.ndarray, output)
