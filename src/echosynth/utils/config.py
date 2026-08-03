"""YAML configuration composition, validation, and provenance."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from echosynth.utils.io import atomic_write_text


@dataclass(frozen=True)
class Config(Mapping[str, Any]):
    """Validated resolved configuration with attribute and mapping access."""

    data: dict[str, Any]
    source: Path

    def __post_init__(self) -> None:
        seed = self.data.get("seed", 2024)
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("config seed must be a non-negative integer")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self.data[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated mutable copy of the resolved values."""
        return copy.deepcopy(self.data)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_mapping(mapping: dict[str, Any], parent: Path, seen: set[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    base_spec = mapping.get("_base")
    if base_spec is not None:
        bases = [base_spec] if isinstance(base_spec, str) else base_spec
        if not isinstance(bases, list) or not all(isinstance(item, str) for item in bases):
            raise ValueError("_base must be a path string or list of path strings")
        for raw in bases:
            candidate = Path(raw)
            base_path = (candidate if candidate.is_absolute() else parent / candidate).resolve()
            if not base_path.exists():
                project_path = (Path.cwd() / candidate).resolve()
                base_path = project_path if project_path.exists() else base_path
            result = _merge(result, _load_yaml(base_path, seen))
    own = {key: value for key, value in mapping.items() if key != "_base"}
    for key, value in own.items():
        if isinstance(value, dict):
            own[key] = _resolve_mapping(value, parent, seen)
    return _merge(result, own)


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance involving {path}")
    seen.add(path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        return _resolve_mapping(loaded, path.parent, seen)
    finally:
        seen.remove(path)


def _set_nested(values: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"invalid override key: {dotted_key}")
    cursor = values
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"override traverses scalar at {key}")
        cursor = child
    cursor[keys[-1]] = value


def load_config(path: Path | str, overrides: list[str] | None = None) -> Config:
    """Load YAML with recursive ``_base`` inheritance and ``a.b=c`` overrides."""
    source = Path(path).resolve()
    values = _load_yaml(source, set())
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must have form a.b=c: {override}")
        key, raw = override.split("=", 1)
        _set_nested(values, key, yaml.safe_load(raw))
    return Config(data=values, source=source)


def git_sha(root: Path | None = None) -> str:
    """Return the current Git SHA, or ``unavailable`` outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def save_resolved(config: Config, out_dir: Path) -> None:
    """Persist the resolved YAML and Git SHA beside run artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "config.yaml", yaml.safe_dump(config.to_dict(), sort_keys=True))
    atomic_write_text(out_dir / "git_sha.txt", git_sha(config.source.parent) + "\n")
