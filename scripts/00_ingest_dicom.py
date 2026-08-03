"""CLI for DICOM ingest."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.data.dicom_ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/duke.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = ingest(
        args.root,
        args.out_dir,
        args.manifest,
        args.workers,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    print(f"accepted={len(result)} rejections={result.attrs['rejections']}")


if __name__ == "__main__":
    main()
