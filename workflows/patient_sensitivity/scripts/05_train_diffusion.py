"""CLI for leakage-safe class-conditional diffusion training."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.generative.trainer import train_diffusion
from echosynth.utils.config import load_config, save_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs/diffusion/ckpt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    resolved = load_config(args.config, args.overrides)
    config = resolved.to_dict()
    manifest = args.manifest or Path(
        config.get("data", {}).get("manifest", config.get("manifest", ""))
    )
    class_map = config.get("classes") or config.get("generator", {}).get("classes")
    if not class_map or not str(manifest):
        parser.error("config must resolve a manifest and classes mapping")
    checkpoint = train_diffusion(
        Path(manifest),
        args.out_dir,
        class_map,
        image_size=int(config.get("image_size", 112)),
        steps=args.steps,
        batch_size=int(config.get("batch_size", 16)),
        lr=float(config.get("lr", 8e-5)),
        seed=int(config.get("seed", 2024)),
        model_dim=int(config.get("dim", 64)),
        dim_mults=tuple(config.get("dim_mults", (1, 2, 4, 8))),
        full_attn=tuple(config.get("full_attn", (False, False, False, True))),
        timesteps=int(config.get("timesteps", 1000)),
        sampling_timesteps=int(config.get("sampling_timesteps", 500)),
    )
    save_resolved(resolved, args.out_dir.parent)
    print(checkpoint)


if __name__ == "__main__":
    main()
