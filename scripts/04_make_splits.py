"""CLI for deterministic unique-patient splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echosynth.data.manifest import write_manifest
from echosynth.data.splits import make_patient_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    result = make_patient_splits(frame, seed=args.seed)
    write_manifest(result, args.manifest)
    print(result.groupby(["group", "split"]).size())


if __name__ == "__main__":
    main()
