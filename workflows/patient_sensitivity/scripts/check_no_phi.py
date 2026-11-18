"""Reject staged research data and PHI-like filenames/content."""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_SUFFIXES = {".dcm", ".png", ".pt", ".h5"}
PHI_PATTERN = re.compile(r"(?<!\d)\d{8,}(?!\d)|\d{4}-\d{2}-\d{2}")


def main(paths: list[str]) -> int:
    """Return nonzero when any candidate could contain PHI or research data."""
    rejected: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.parts[:1] == ("data",) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            rejected.append(f"{raw}: forbidden research-data path or type")
            continue
        if PHI_PATTERN.search(path.name):
            rejected.append(f"{raw}: PHI-like filename")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if re.search(r"(?<!\d)\d{8,}(?!\d)", text):
            rejected.append(f"{raw}: MRN-like digit sequence in content")
    if rejected:
        print("Commit rejected:\n" + "\n".join(rejected), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
