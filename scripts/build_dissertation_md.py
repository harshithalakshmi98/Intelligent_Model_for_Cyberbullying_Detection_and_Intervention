#!/usr/bin/env python3
"""Export dissertation markdown to outputs/reports (same content as the PDF)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "research" / "Dissertation.md"
TARGET = ROOT / "outputs" / "reports" / "F21MP_Dissertation.md"


def main() -> int:
    if not SOURCE.exists():
        print(f"Missing {SOURCE}", file=sys.stderr)
        return 1

    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace("../../outputs/media/", "../media/")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
