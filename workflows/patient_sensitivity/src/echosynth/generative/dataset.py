"""Training-only labeled echo dataset for diffusion."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from echosynth.data.manifest import LeakageError, read_manifest


class LabeledEchoDataset(Dataset[tuple[torch.Tensor, int]]):
    """Return ``(image_tensor, class_idx)`` from an all-training manifest."""

    def __init__(
        self,
        manifest: Path,
        split: str,
        label_col: str,
        class_map: dict[str, int],
        image_size: int = 112,
        augment_horizontal_flip: bool = True,
    ) -> None:
        if split != "train":
            raise LeakageError("Diffusion dataset accepts split='train' only")
        frame = read_manifest(manifest)
        non_train = (
            frame.loc[frame["split"].astype(str) != "train", "image_id"].astype(str).tolist()
        )
        if non_train:
            raise LeakageError(f"Generator manifest contains non-train rows: {non_train[:20]}")
        if label_col not in frame:
            raise ValueError(f"missing label column: {label_col}")
        unknown = sorted(set(frame[label_col].astype(str)) - set(class_map))
        if unknown:
            raise ValueError(f"labels absent from class_map: {unknown}")
        self.frame = frame.reset_index(drop=True)
        self.label_col = label_col
        self.class_map = class_map
        self.image_size = image_size
        self.augment_horizontal_flip = augment_horizontal_flip

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(str(row["png_path"])) as image:
            grayscale = image.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(grayscale, dtype=np.float32).copy() / 127.5 - 1.0
        if self.augment_horizontal_flip and random.random() < 0.5:
            array = np.fliplr(array).copy()
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, self.class_map[str(row[self.label_col])]
