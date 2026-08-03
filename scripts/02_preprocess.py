"""CLI for registration, masking, and 112-pixel preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.data.preprocess import preprocess_manifest
from echosynth.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-reference", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config).to_dict()
    settings = config.get("preprocess", {})
    preprocess_manifest(
        args.manifest,
        args.out_dir,
        Path(settings.get("reference", "results/reference_sector.json")),
        build_reference_first=args.build_reference,
        qa_path=Path(settings.get("qa_sheet", "results/preprocess_qa.png")),
        image_size=int(config.get("image_size", 112)),
    )


if __name__ == "__main__":
    main()
