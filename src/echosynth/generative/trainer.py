"""Accelerate/AMP diffusion trainer with EMA, checkpoints, and provenance."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from echosynth.generative.dataset import LabeledEchoDataset
from echosynth.generative.diffusion import GaussianDiffusion
from echosynth.generative.unet_cfg import ClassConditionalUnet
from echosynth.utils.checkpoint import save_checkpoint
from echosynth.utils.seed import set_seed


def train_diffusion(
    manifest: Path,
    out_dir: Path,
    class_map: dict[str, int],
    *,
    label_col: str = "group",
    image_size: int = 112,
    steps: int = 1000,
    batch_size: int = 16,
    lr: float = 8e-5,
    seed: int = 2024,
    model_dim: int = 64,
    dim_mults: tuple[int, ...] = (1, 2, 4, 8),
    full_attn: tuple[bool, ...] = (False, False, False, True),
    timesteps: int = 1000,
    sampling_timesteps: int = 500,
    save_and_sample_every: int = 1000,
    mixed_precision: str | None = None,
) -> Path:
    """Train with EMA/AMP on a leakage-checked training-only manifest."""
    if steps < 1:
        raise ValueError("steps must be positive")
    set_seed(seed)
    dataset = LabeledEchoDataset(manifest, "train", label_col, class_map, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    precision = mixed_precision or ("fp16" if torch.cuda.is_available() else "no")
    accelerator = Accelerator(mixed_precision=precision)
    model = ClassConditionalUnet(
        dim=model_dim,
        dim_mults=dim_mults,
        channels=1,
        num_classes=len(class_map),
        full_attn=full_attn,
        attn_dim_head=max(4, model_dim),
        attn_heads=1 if model_dim < 16 else 4,
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
    )
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    diffusion, optimizer, loader = accelerator.prepare(diffusion, optimizer, loader)
    unwrapped = accelerator.unwrap_model(diffusion)
    ema = EMA(unwrapped.model, beta=0.995, update_after_step=0, update_every=1).to(
        accelerator.device
    )
    cycle = itertools.cycle(loader)
    final_loss = float("nan")
    for step in range(1, steps + 1):
        images, classes = next(cycle)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss = diffusion(images, classes)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)
        optimizer.step()
        ema.update()
        final_loss = float(loss.detach().cpu())
        if step % save_and_sample_every == 0:
            accelerator.wait_for_everyone()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "ema_best.pt"
    state: dict[str, Any] = {
        "model": unwrapped.model.state_dict(),
        "ema": ema.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    metadata: dict[str, Any] = {
        "model_config": unwrapped.model.model_config,
        "diffusion_config": unwrapped.diffusion_config,
        "class_map": class_map,
        "seed": seed,
        "steps": steps,
        "final_loss": final_loss,
    }
    if accelerator.is_main_process:
        save_checkpoint(checkpoint, state, metadata)
    return checkpoint
