"""Exact deterministic encounter splitting tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from echosynth.data.splits import make_encounter_splits
from echosynth.eval.guard import TestSetGuard, TestSetReuseError


def _duke_encounters() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group, count in {"CA": 309, "GEN": 2203, "LVH": 2185}.items():
        for index in range(count):
            encounter = f"{group.lower()}-{index}"
            rows.append(
                {
                    "image_id": f"image-{encounter}",
                    "patient_id": f"patient-{encounter}",
                    "encounter_id": encounter,
                    "group": group,
                    "png_path": f"/derived/{encounter}.png",
                }
            )
    return pd.DataFrame(rows)


def test_table_one_counts_and_determinism() -> None:
    source = _duke_encounters()
    first = make_encounter_splits(source, seed=2024)
    second = make_encounter_splits(source, seed=2024)
    different = make_encounter_splits(source, seed=7)
    counts = first.groupby(["group", "split"]).size().to_dict()
    assert counts == {
        ("CA", "test"): 47,
        ("CA", "train"): 216,
        ("CA", "val"): 46,
        ("GEN", "test"): 331,
        ("GEN", "train"): 1542,
        ("GEN", "val"): 330,
        ("LVH", "test"): 328,
        ("LVH", "train"): 1529,
        ("LVH", "val"): 328,
    }
    pd.testing.assert_series_equal(first["split"], second["split"])
    assert not first["split"].equals(different["split"])


def test_images_from_encounter_stay_together() -> None:
    source = _duke_encounters().iloc[:20].copy()
    source = pd.concat([source, source.iloc[:2].assign(image_id=["extra-a", "extra-b"])])
    split = make_encounter_splits(source)
    assert split.groupby("encounter_id")["split"].nunique().max() == 1


def test_guard_raises_on_second_entry(tmp_path: Path) -> None:
    guard = TestSetGuard(tmp_path / "test_access.jsonl")
    with guard:
        pass
    with pytest.raises(TestSetReuseError):
        with guard:
            pass
    assert "test_set_access" in (tmp_path / "test_access.jsonl").read_text(encoding="utf-8")
