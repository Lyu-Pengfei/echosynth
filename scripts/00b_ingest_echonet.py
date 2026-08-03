"""CLI for EchoNet-Dynamic labels or EchoNet-LVH phenotype manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echosynth.data.echonet import adapt_dynamic, adapt_lvh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dynamic", "lvh"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-official", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    if args.dataset == "dynamic":
        result = adapt_dynamic(frame, validate_expected=args.validate_official)
    else:
        result, threshold = adapt_lvh(frame, validate_official=args.validate_official)
        print(f"train_only_tau={threshold:.6f}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.manifest, index=False)
    print(f"rows={len(result)}")


if __name__ == "__main__":
    main()
