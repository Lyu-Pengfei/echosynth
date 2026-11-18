"""CLI for validation-selected, single-test-access experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.classify.paradigms import run_experiment
from echosynth.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    result = run_experiment(load_config(args.config, args.overrides))
    print(f"run={result.run_id} selected={result.selected_synthetic_count} dir={result.run_dir}")


if __name__ == "__main__":
    main()
