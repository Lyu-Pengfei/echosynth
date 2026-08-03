"""CLI for deterministic conditional DDIM sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from echosynth.generative.sample import generate
from echosynth.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--counts", required=True, help="YAML mapping, for example '{CA: 100}'")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides).to_dict()
    counts = yaml.safe_load(args.counts)
    if not isinstance(counts, dict):
        parser.error("--counts must be a YAML mapping")
    result = generate(
        args.ckpt,
        {str(key): int(value) for key, value in counts.items()},
        float(config.get("cond_scale", 2.0)),
        int(config.get("sampling_timesteps", 500)),
        args.out_dir,
        int(config.get("seed", 2024)),
        args.manifest_out,
    )
    print(f"generated={len(result)} sampler=DDIM")


if __name__ == "__main__":
    main()
