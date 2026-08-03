"""Persistent single-access test-set guard (spec section 4.7)."""

from __future__ import annotations

import inspect
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

from echosynth.utils.io import atomic_write_text


class TestSetReuseError(RuntimeError):
    """Raised when a run attempts a second test-set evaluation."""

    __test__ = False


class TestSetGuard:
    """Permit exactly one audited test-set evaluation for a run."""

    _lock = threading.Lock()
    __test__ = False

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._entered = False

    def __enter__(self) -> TestSetGuard:
        with self._lock:
            if self._entered or (self.log_path.exists() and self.log_path.stat().st_size > 0):
                raise TestSetReuseError(f"Test set already evaluated; audit log: {self.log_path}")
            caller = inspect.stack()[1]
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "caller": f"{caller.filename}:{caller.lineno}:{caller.function}",
                "event": "test_set_access",
            }
            atomic_write_text(self.log_path, json.dumps(event, sort_keys=True) + "\n")
            self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False
