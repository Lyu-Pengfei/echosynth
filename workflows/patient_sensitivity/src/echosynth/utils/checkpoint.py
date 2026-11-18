"""Model checkpoint serialization with provenance metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Atomically save state and required metadata to a PyTorch checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state": state, "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and validate its top-level structure."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"state", "metadata"}:
        raise ValueError(f"Invalid checkpoint structure: {path}")
    state = payload["state"]
    metadata = payload["metadata"]
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError(f"Invalid checkpoint payload types: {path}")
    return state, metadata
