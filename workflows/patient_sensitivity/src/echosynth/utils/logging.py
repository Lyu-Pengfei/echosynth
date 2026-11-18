"""Structured logging and rejection accounting."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class RejectionCounter(Mapping[str, int]):
    """Tally named rejection reasons and render a deterministic summary table."""

    _counts: Counter[str] = field(default_factory=Counter)

    def add(self, reason: str, count: int = 1) -> None:
        """Increment a non-empty reason code."""
        if not reason or count < 0:
            raise ValueError("reason must be non-empty and count non-negative")
        self._counts[reason] += count

    def __getitem__(self, key: str) -> int:
        return self._counts[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._counts))

    def __len__(self) -> int:
        return len(self._counts)

    def render(self) -> str:
        """Return a plain-text two-column summary table."""
        rows = ["reason | count", "--- | ---"]
        rows.extend(f"{reason} | {self._counts[reason]}" for reason in sorted(self._counts))
        rows.append(f"TOTAL | {sum(self._counts.values())}")
        return "\n".join(rows)

    def log_summary(self, logger: logging.Logger) -> None:
        """Log the complete rejection summary without hiding empty runs."""
        logger.info("Rejection summary:\n%s", self.render())
