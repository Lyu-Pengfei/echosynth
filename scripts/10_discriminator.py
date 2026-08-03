"""CLI for patient-separated real-versus-synthetic discrimination."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from echosynth.eval.discriminator import make_patient_splits, run_discriminator
from echosynth.utils.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if "split" not in frame:
        frame = make_patient_splits(frame)
    result = run_discriminator(frame, args.out_dir, epochs=args.epochs)
    atomic_write_json(args.out_dir / "metrics.json", asdict(result))
    print(asdict(result))


if __name__ == "__main__":
    main()
