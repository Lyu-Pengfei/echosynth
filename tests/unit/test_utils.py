"""Fast tests for P1 utility infrastructure."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from echosynth.utils.checkpoint import load_checkpoint, save_checkpoint
from echosynth.utils.config import load_config, save_resolved
from echosynth.utils.io import atomic_write_json, atomic_write_text
from echosynth.utils.logging import RejectionCounter
from echosynth.utils.seed import set_seed


def test_seed_is_reproducible() -> None:
    set_seed(7)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    set_seed(7)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


def test_config_inheritance_override_and_save(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("seed: 4\nmodel: {width: 8, depth: 2}\n", encoding="utf-8")
    child.write_text("_base: base.yaml\nmodel: {depth: 3}\n", encoding="utf-8")
    config = load_config(child, ["model.width=16", "flag=true"])
    assert config["model"] == {"width": 16, "depth": 3}
    assert config["flag"] is True
    save_resolved(config, tmp_path / "run")
    saved = yaml.safe_load((tmp_path / "run/config.yaml").read_text(encoding="utf-8"))
    assert saved == config.to_dict()
    assert (tmp_path / "run/git_sha.txt").is_file()


def test_atomic_writes_and_checkpoint(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "nested/text.txt", "hello")
    atomic_write_json(tmp_path / "nested/value.json", {"b": 1})
    assert json.loads((tmp_path / "nested/value.json").read_text(encoding="utf-8")) == {"b": 1}
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, {"weights": torch.ones(2)}, {"seed": 3})
    state, metadata = load_checkpoint(checkpoint)
    assert torch.equal(state["weights"], torch.ones(2))
    assert metadata == {"seed": 3}


def test_rejection_counter() -> None:
    counter = RejectionCounter()
    counter.add("UNREADABLE")
    counter.add("UNREADABLE", 2)
    counter.add("COLOR_DOPPLER")
    assert counter["UNREADABLE"] == 3
    assert "TOTAL | 4" in counter.render()
    with pytest.raises(ValueError):
        counter.add("")
