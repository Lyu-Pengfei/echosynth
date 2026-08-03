"""Item-wise descriptive expert-survey analysis (paper section 2.8)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalized_entropy(values: Sequence[Any], categories: Sequence[Any] | None = None) -> float:
    """Return base-2 response entropy divided by log2(K) for this item."""
    observed = pd.Series(values).dropna()
    if observed.empty:
        return float("nan")
    category_values = list(categories) if categories is not None else observed.unique().tolist()
    category_count = len(category_values)
    if category_count <= 1:
        return 0.0
    probabilities = observed.value_counts(normalize=True)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return entropy / math.log2(category_count)


def recode_paired_preference(
    chosen_side: str, real_side: str, *, cannot_decide: str = "cannot_decide"
) -> str:
    """Recode blinded left/right choice after unblinding."""
    choice = str(chosen_side).strip().lower()
    if choice == cannot_decide:
        return "cannot_decide"
    if choice not in {"left", "right"} or real_side not in {"left", "right"}:
        raise ValueError("paired sides must be left/right or cannot_decide")
    return "prefers_real" if choice == real_side else "prefers_synthetic"


def analyze_survey(
    frame: pd.DataFrame,
    item_columns: Sequence[str],
    *,
    source_col: str = "source",
    categories: Mapping[str, Sequence[Any]] | None = None,
    confirmatory_test: bool = False,
) -> pd.DataFrame:
    """Emit item-wise percentages/denominators and group-normalized entropy only."""
    if confirmatory_test:
        raise NotImplementedError(
            "Responses are repeated within evaluator; v1 emits descriptive statistics only"
        )
    missing = (set(item_columns) | {source_col}) - set(frame)
    if missing:
        raise ValueError(f"survey frame missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for source, group in frame.groupby(source_col, observed=True):
        for item in item_columns:
            responses = group[item].dropna()
            denominator = len(responses)
            entropy = normalized_entropy(
                responses,
                None if categories is None else categories.get(item),
            )
            for response, count in responses.value_counts(dropna=False).items():
                records.append(
                    {
                        "source": source,
                        "item": item,
                        "response": response,
                        "count": int(count),
                        "n": denominator,
                        "percentage": 100 * int(count) / denominator,
                        "normalized_entropy": entropy,
                    }
                )
    table = pd.DataFrame(records)
    if not table.empty:
        group_means = (
            table.drop_duplicates(["source", "item"])
            .groupby("source", observed=True)["normalized_entropy"]
            .mean()
        )
        table["mean_normalized_entropy"] = table["source"].map(group_means)
    return table


def write_survey_tables(table: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    """Write the same tidy descriptive table as CSV and formatted Markdown."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    markdown_path.write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
