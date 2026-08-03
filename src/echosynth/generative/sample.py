"""Deterministic conditional diffusion sampling into a physically separate tree."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from echosynth.generative.diffusion import GaussianDiffusion
from echosynth.generative.unet_cfg import ClassConditionalUnet
from echosynth.utils.checkpoint import load_checkpoint
from echosynth.utils.io import atomic_write_bytes, atomic_write_text
from echosynth.utils.seed import set_seed


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor.detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array.squeeze(0), mode="L").save(buffer, "PNG", optimize=False)
    atomic_write_bytes(path, buffer.getvalue())


def generate(
    ckpt: Path,
    n_per_class: dict[str, int],
    cond_scale: float,
    sampling_steps: int,
    out_dir: Path,
    seed: int,
    manifest_out: Path,
) -> pd.DataFrame:
    """Write reproducible PNGs and a synthetic manifest whose split is always training."""
    state, metadata = load_checkpoint(ckpt)
    model_config = dict(metadata["model_config"])
    diffusion_config = dict(metadata["diffusion_config"])
    class_map = dict(metadata["class_map"])
    unknown = sorted(set(n_per_class) - set(class_map))
    if unknown:
        raise ValueError(f"requested unknown classes: {unknown}")
    if any(count < 0 for count in n_per_class.values()):
        raise ValueError("sample counts must be non-negative")
    diffusion_config["sampling_timesteps"] = sampling_steps
    model = ClassConditionalUnet(**model_config)
    model.load_state_dict(state.get("ema", state["model"]))
    diffusion = GaussianDiffusion(model, **diffusion_config)
    diffusion.eval()
    set_seed(seed)
    run_id = hashlib.sha256(f"{ckpt.resolve()}:{seed}".encode()).hexdigest()[:16]
    rows: list[dict[str, Any]] = []
    for label, count in n_per_class.items():
        if count == 0:
            continue
        classes = torch.full((count,), int(class_map[label]), dtype=torch.long)
        samples = diffusion.sample(classes, cond_scale)
        class_dir = out_dir / label
        for index, sample in enumerate(samples):
            image_id = hashlib.sha256(f"{run_id}:{label}:{index}".encode()).hexdigest()[:24]
            destination = class_dir / f"{image_id}.png"
            _save_image(sample, destination)
            rows.append(
                {
                    "image_id": image_id,
                    "task": str(metadata.get("task", "conditional_echo")),
                    "class_label": label,
                    "guidance_scale": cond_scale,
                    "sampling_steps": sampling_steps,
                    "generator_run_id": run_id,
                    "png_path": str(destination),
                    "seed": seed,
                    "split": "train",
                }
            )
    frame = pd.DataFrame(rows)
    atomic_write_text(manifest_out, frame.to_csv(index=False))
    return frame
