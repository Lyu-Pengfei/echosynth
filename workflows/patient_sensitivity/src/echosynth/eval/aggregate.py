"""Image-to-encounter prediction aggregation."""

from __future__ import annotations

import pandas as pd


def to_encounter_level(
    df: pd.DataFrame, prob_col: str = "p", by: str = "encounter_id"
) -> pd.DataFrame:
    """Mean image probabilities within encounter, the only reported unit.

    Implements paper lines 240-244. Reporting at image level would violate the independence
    assumption because multiple images from one encounter are correlated.
    """
    missing = {by, prob_col} - set(df)
    if missing:
        raise ValueError(f"aggregation frame missing columns: {sorted(missing)}")
    label_columns = [column for column in ("label", "y", "target") if column in df]
    for column in label_columns:
        counts = df.groupby(by, observed=True)[column].nunique(dropna=False)
        offenders = sorted(counts[counts > 1].index.astype(str).tolist())
        if offenders:
            raise ValueError(f"inconsistent {column} within encounters: {offenders[:20]}")
    aggregations: dict[str, str] = {prob_col: "mean"}
    aggregations.update({column: "first" for column in label_columns})
    for column in ("patient_id", "group", "split"):
        if column in df and column != by:
            aggregations[column] = "first"
    return df.groupby(by, as_index=False, observed=True).agg(aggregations)
