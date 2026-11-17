"""Deterministic unique-patient splitting tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from echosynth.data.manifest import LeakageError
from echosynth.data.splits import make_patient_splits
from echosynth.eval.guard import TestSetGuard, TestSetReuseError


def _cohort() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group, count in {"CA": 30, "GEN": 60, "LVH": 50}.items():
        for index in range(count):
            patient = f"patient-{group.lower()}-{index}"
            for visit in range(1 + (index % 3 == 0)):
                encounter = f"{group.lower()}-{index}-{visit}"
                rows.append(
                    {
                        "image_id": f"image-{encounter}",
                        "patient_id": patient,
                        "encounter_id": encounter,
                        "group": group,
                        "png_path": f"/derived/{encounter}.png",
                    }
                )
    return pd.DataFrame(rows)


def test_table_one_counts_and_determinism() -> None:
    source = _cohort()
    first = make_patient_splits(source, seed=2024)
    second = make_patient_splits(source, seed=2024)
    different = make_patient_splits(source, seed=7)
    counts = (
        first[["patient_id", "group", "split"]]
        .drop_duplicates()
        .groupby(["group", "split"])
        .size()
        .to_dict()
    )
    assert counts == {
        ("CA", "test"): 5,
        ("CA", "train"): 21,
        ("CA", "val"): 4,
        ("GEN", "test"): 9,
        ("GEN", "train"): 42,
        ("GEN", "val"): 9,
        ("LVH", "test"): 8,
        ("LVH", "train"): 35,
        ("LVH", "val"): 7,
    }
    pd.testing.assert_series_equal(first["split"], second["split"])
    assert not first["split"].equals(different["split"])


def test_all_encounters_from_patient_stay_together() -> None:
    source = _cohort()
    split = make_patient_splits(source)
    assert split.groupby("patient_id")["split"].nunique().max() == 1
    assert split.groupby("encounter_id")["split"].nunique().max() == 1


def test_conflicting_patient_group_raises() -> None:
    source = _cohort().iloc[:2].copy()
    source.loc[source.index[1], "group"] = "LVH"
    with pytest.raises(LeakageError, match="patient_id has inconsistent group"):
        make_patient_splits(source)


def test_guard_raises_on_second_entry(tmp_path: Path) -> None:
    guard = TestSetGuard(tmp_path / "test_access.jsonl")
    with guard:
        pass
    with pytest.raises(TestSetReuseError):
        with guard:
            pass
    assert "test_set_access" in (tmp_path / "test_access.jsonl").read_text(encoding="utf-8")
