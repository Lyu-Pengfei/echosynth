"""Verify that the repository hygiene hook rejects prohibited inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_hook() -> object:
    path = Path(__file__).parents[2] / "scripts/check_no_phi.py"
    spec = importlib.util.spec_from_file_location("check_no_phi", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_phi_hook(tmp_path: Path, monkeypatch: object) -> None:
    module = _load_hook()
    safe = tmp_path / "notes.txt"
    safe.write_text("deidentified text", encoding="utf-8")
    phi = tmp_path / ("patient_" + "1234" + "5678.txt")
    phi.write_text("unsafe", encoding="utf-8")
    dated = tmp_path / "scan_2020-01-01.txt"
    dated.write_text("unsafe", encoding="utf-8")
    assert module.main([str(safe)]) == 0  # type: ignore[attr-defined]
    assert module.main([str(phi)]) == 1  # type: ignore[attr-defined]
    assert module.main([str(dated)]) == 1  # type: ignore[attr-defined]
