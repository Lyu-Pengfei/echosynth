"""CLI for descriptive, item-wise expert-survey summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echosynth.eval.survey import analyze_survey, write_survey_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    table = analyze_survey(pd.read_csv(args.responses), args.items, source_col=args.source_col)
    write_survey_tables(table, args.out_dir / "survey.csv", args.out_dir / "survey.md")
    print(f"rows={len(table)}")


if __name__ == "__main__":
    main()
