"""Atomic filesystem I/O helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace *path* atomically with *payload*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, payload: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write stable, human-readable JSON."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
