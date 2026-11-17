"""Image-to-encounter and encounter-to-patient prediction aggregation."""

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


def to_patient_level(df: pd.DataFrame, prob_col: str = "p") -> pd.DataFrame:
    """Average images within encounter, then encounters within unique patient.

    The two-stage mean prevents encounters with more exported images from receiving
    disproportionate weight in the patient-level endpoint.
    """
    required = {"encounter_id", "patient_id", prob_col}
    missing = required - set(df)
    if missing:
        raise ValueError(f"patient aggregation frame missing columns: {sorted(missing)}")
    encounter = to_encounter_level(df, prob_col=prob_col)
    for column in ("label", "y", "target", "group", "split"):
        if column in encounter:
            counts = encounter.groupby("patient_id", observed=True)[column].nunique(dropna=False)
            offenders = sorted(counts[counts > 1].index.astype(str).tolist())
            if offenders:
                raise ValueError(f"inconsistent {column} within patients: {offenders[:20]}")
    aggregations: dict[str, str] = {prob_col: "mean", "encounter_id": "nunique"}
    for column in ("label", "y", "target", "group", "split"):
        if column in encounter:
            aggregations[column] = "first"
    patient = encounter.groupby("patient_id", as_index=False, observed=True).agg(aggregations)
    return patient.rename(columns={"encounter_id": "n_encounters"})
