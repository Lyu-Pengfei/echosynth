"""Structurally separate training and evaluation image transforms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from PIL import Image, ImageEnhance


class ImageTransform(Protocol):
    """Callable grayscale transform contract."""

    training: bool

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor: ...


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("L")
    return Image.fromarray(np.asarray(image).astype(np.uint8), mode="L")


def _tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0)


@dataclass(frozen=True)
class EvalTransform:
    """Deterministic resize and normalization with no augmentation methods."""

    image_size: int = 112
    training: bool = False

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        return _tensor(_to_pil(image), self.image_size)


@dataclass(frozen=True)
class TrainTransform:
    """Paper lines 208-210 training-only flip, rotation, brightness, and contrast."""

    image_size: int = 112
    flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    training: bool = True

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        output = _to_pil(image)
        if random.random() < self.flip_probability:
            output = output.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        output = output.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        output = ImageEnhance.Brightness(output).enhance(
            random.uniform(1 - self.brightness, 1 + self.brightness)
        )
        output = ImageEnhance.Contrast(output).enhance(
            random.uniform(1 - self.contrast, 1 + self.contrast)
        )
        return _tensor(output, self.image_size)


def build_transform(training: bool, image_size: int = 112) -> ImageTransform:
    """Return a distinct type so train augmentation cannot run in evaluation."""
    transform = (
        TrainTransform(image_size=image_size) if training else EvalTransform(image_size=image_size)
    )
    return cast(ImageTransform, transform)
