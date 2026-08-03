"""CLI for pooled-histogram fidelity metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echosynth.eval.fidelity import evaluate_fidelity
from echosynth.utils.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    real = pd.read_csv(args.real_manifest)
    synthetic = pd.read_csv(args.synthetic_manifest)
    metrics = evaluate_fidelity(
        [Path(path) for path in real["png_path"].astype(str)],
        [Path(path) for path in synthetic["png_path"].astype(str)],
    )
    atomic_write_json(args.output, metrics.to_dict())
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
