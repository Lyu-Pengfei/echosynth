"""Classifier-free-guidance conditional U-Net based on the MIT lucidrains structure."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from einops import rearrange
from torch import nn


def _groups(channels: int, requested: int) -> int:
    return math.gcd(channels, requested)


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal time encoding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=time.device) * -scale)
        embedding = time.float()[:, None] * frequencies[None, :]
        encoded = torch.cat((embedding.sin(), embedding.cos()), dim=1)
        return nn.functional.pad(encoded, (0, self.dim - encoded.shape[1]))


class ResBlock(nn.Module):
    """Two-convolution residual block modulated by time/class embedding."""

    def __init__(self, input_dim: int, output_dim: int, time_dim: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(input_dim, groups), input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(output_dim, groups), output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, output_dim * 2))
        self.residual = (
            nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(nn.functional.silu(self.norm1(x)))
        scale, shift = self.time(embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        hidden = self.conv2(nn.functional.silu(hidden))
        return cast(torch.Tensor, hidden + self.residual(x))


class Attention(nn.Module):
    """Spatial multi-head self-attention at configured deeper resolutions."""

    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        hidden = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.GroupNorm(1, dim)
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.output = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (
            rearrange(item, "b (h d) x y -> b h (x y) d", h=self.heads) for item in (q, k, v)
        )
        attention = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.matmul(attention, v)
        output = rearrange(output, "b h (x y) d -> b (h d) x y", x=x.shape[-2], y=x.shape[-1])
        return cast(torch.Tensor, x + self.output(output))


class ClassConditionalUnet(nn.Module):
    """Class-conditional denoising U-Net implementing paper lines 184-192."""

    def __init__(
        self,
        dim: int = 64,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 1,
        num_classes: int = 3,
        cond_drop_prob: float = 0.1,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 16,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        full_attn: Sequence[bool] = (False, False, False, True),
    ) -> None:
        super().__init__()
        if len(dim_mults) != len(full_attn):
            raise ValueError("dim_mults and full_attn must have equal length")
        if num_classes < 2 or not 0 <= cond_drop_prob <= 1:
            raise ValueError("num_classes >= 2 and cond_drop_prob in [0, 1] are required")
        self.channels = channels
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        self.model_config = {
            "dim": dim,
            "dim_mults": tuple(dim_mults),
            "channels": channels,
            "num_classes": num_classes,
            "cond_drop_prob": cond_drop_prob,
            "resnet_block_groups": resnet_block_groups,
            "learned_sinusoidal_dim": learned_sinusoidal_dim,
            "attn_dim_head": attn_dim_head,
            "attn_heads": attn_heads,
            "full_attn": tuple(full_attn),
        }
        time_dim = dim * 4
        self.initial = nn.Conv2d(channels, dim, 7, padding=3)
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, time_dim)
        self.null_class_embedding = nn.Parameter(torch.randn(time_dim) * 0.02)
        dimensions = [dim * multiplier for multiplier in dim_mults]
        self.downs = nn.ModuleList()
        current = dim
        for index, output_dim in enumerate(dimensions):
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(current, output_dim, time_dim, resnet_block_groups),
                        ResBlock(output_dim, output_dim, time_dim, resnet_block_groups),
                        Attention(output_dim, attn_heads, attn_dim_head)
                        if full_attn[index]
                        else nn.Identity(),
                        nn.Conv2d(output_dim, output_dim, 4, 2, 1)
                        if index < len(dimensions) - 1
                        else nn.Identity(),
                    ]
                )
            )
            current = output_dim
        self.mid1 = ResBlock(current, current, time_dim, resnet_block_groups)
        self.mid_attention = (
            Attention(current, attn_heads, attn_dim_head) if any(full_attn) else nn.Identity()
        )
        self.mid2 = ResBlock(current, current, time_dim, resnet_block_groups)
        self.ups = nn.ModuleList()
        for index in reversed(range(len(dimensions))):
            output_dim = dimensions[index]
            next_dim = dimensions[index - 1] if index > 0 else dimensions[0]
            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(output_dim * 2, output_dim, time_dim, resnet_block_groups),
                        ResBlock(output_dim, output_dim, time_dim, resnet_block_groups),
                        Attention(output_dim, attn_heads, attn_dim_head)
                        if full_attn[index]
                        else nn.Identity(),
                        nn.ConvTranspose2d(output_dim, next_dim, 4, 2, 1)
                        if index > 0
                        else nn.Identity(),
                    ]
                )
            )
        self.final = nn.Sequential(
            nn.GroupNorm(_groups(dimensions[0], resnet_block_groups), dimensions[0]),
            nn.SiLU(),
            nn.Conv2d(dimensions[0], channels, 3, padding=1),
        )

    def _conditioning(
        self, time: torch.Tensor, classes: torch.Tensor, cond_drop_prob: float
    ) -> torch.Tensor:
        if classes.ndim != 1 or len(classes) != len(time):
            raise ValueError("classes and time must be one-dimensional and batch-aligned")
        if torch.any((classes < 0) | (classes >= self.num_classes)):
            raise ValueError("class index outside configured range")
        conditional = self.class_embedding(classes)
        if cond_drop_prob == 0:
            class_embedding = conditional
        elif cond_drop_prob == 1:
            class_embedding = self.null_class_embedding.expand_as(conditional)
        else:
            drop = torch.rand(len(classes), device=classes.device) < cond_drop_prob
            null = self.null_class_embedding.expand_as(conditional)
            class_embedding = torch.where(drop[:, None], null, conditional)
        return cast(torch.Tensor, self.time_mlp(time) + class_embedding)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: torch.Tensor,
        cond_drop_prob: float | None = None,
    ) -> torch.Tensor:
        """Predict noise with optional classifier-free label dropout."""
        probability = self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        if not 0 <= probability <= 1:
            raise ValueError("cond_drop_prob must be in [0, 1]")
        embedding = self._conditioning(time, classes, probability)
        hidden = self.initial(x)
        skips: list[torch.Tensor] = []
        for block1, block2, attention, downsample in self.downs:
            hidden = block1(hidden, embedding)
            hidden = block2(hidden, embedding)
            hidden = attention(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.mid2(self.mid_attention(self.mid1(hidden, embedding)), embedding)
        for block1, block2, attention, upsample in self.ups:
            hidden = torch.cat((hidden, skips.pop()), dim=1)
            hidden = block1(hidden, embedding)
            hidden = block2(hidden, embedding)
            hidden = attention(hidden)
            hidden = upsample(hidden)
        return cast(torch.Tensor, self.final(hidden))

    def forward_with_cond_scale(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: torch.Tensor,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply epsilon_null + s(epsilon_class - epsilon_null), with s >= 1."""
        if cond_scale < 1:
            raise ValueError("cond_scale must be at least 1")
        conditional = self.forward(x, time, classes, cond_drop_prob=0.0)
        if cond_scale == 1:
            return conditional
        unconditional = self.forward(x, time, classes, cond_drop_prob=1.0)
        return unconditional + cond_scale * (conditional - unconditional)
