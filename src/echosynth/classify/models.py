"""Compact image classifier architecture from paper lines 205-207."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


class CompactCNN(nn.Module):
    """Four Conv3x3-ReLU-MaxPool blocks followed by pooled binary logits."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(width <= 0 for width in channels):
            raise ValueError("CompactCNN requires exactly four positive channel widths")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        blocks: list[nn.Module] = []
        current = in_channels
        for width in channels:
            blocks.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            current = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current, num_classes)
        self.model_config = {
            "in_channels": in_channels,
            "channels": tuple(channels),
            "dropout": dropout,
            "num_classes": num_classes,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two-class logits for a grayscale image batch."""
        features = self.pool(self.features(images)).flatten(1)
        return cast(torch.Tensor, self.classifier(self.dropout(features)))
