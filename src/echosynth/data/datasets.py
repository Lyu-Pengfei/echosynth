"""Shared manifest-backed image-classification datasets."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from echosynth.classify.transforms import ImageTransform


class EchoClassificationDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Return image, binary target, and encounter identifier from a frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_col: str,
        label_map: Mapping[str, int],
        transform: ImageTransform,
    ) -> None:
        required = {"png_path", "encounter_id", label_col}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"classification frame missing columns: {sorted(missing)}")
        unknown = sorted(set(frame[label_col].astype(str)) - set(label_map))
        if unknown:
            raise ValueError(f"classification labels absent from map: {unknown}")
        self.frame = frame.reset_index(drop=True).copy()
        self.label_col = label_col
        self.label_map = dict(label_map)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        with Image.open(str(row["png_path"])) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, self.label_map[str(row[self.label_col])], str(row["encounter_id"])
