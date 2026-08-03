"""CLI for artifact-driven tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from echosynth.report.figures import make_figures
from echosynth.report.tables import make_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/report"))
    args = parser.parse_args()
    tables = make_tables(args.runs, args.out_dir)
    figures = make_figures(args.runs, args.out_dir)
    print(f"tables={len(tables)} figures={len(figures)} out={args.out_dir}")


if __name__ == "__main__":
    main()
