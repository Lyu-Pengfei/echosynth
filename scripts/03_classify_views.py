"""CLI for optional eight-view classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.views.view_classifier import MissingWeightsError, classify_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("onnx", "keras"), default="onnx")
    args = parser.parse_args()
    try:
        result = classify_views(args.manifest, args.weights, args.mode)
    except MissingWeightsError as error:
        parser.exit(2, f"error: {error}\n")
    print(f"classified={len(result)}")


if __name__ == "__main__":
    main()
