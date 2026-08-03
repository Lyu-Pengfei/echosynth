"""CLI for the high-sensitivity QC screen and manual-review application."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.data.qc import run_qc
from echosynth.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply-review", type=Path)
    parser.add_argument("--review-queue", type=Path, default=Path("manifests/qc_review_queue.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    qc = config.to_dict().get("qc", {})
    frame = run_qc(
        args.manifest,
        canny=(int(qc.get("canny_low", 50)), int(qc.get("canny_high", 150))),
        review_queue=args.review_queue,
        reviewed=args.apply_review,
    )
    print(f"flagged={int(frame['qc_flag'].sum())} excluded={int(frame['qc_excluded'].sum())}")


if __name__ == "__main__":
    main()
