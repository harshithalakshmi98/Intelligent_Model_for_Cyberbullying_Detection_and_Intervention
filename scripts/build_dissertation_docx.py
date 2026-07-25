#!/usr/bin/env python3
"""Build Word document from dissertation HTML (same content as the PDF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "outputs" / "reports" / "F21MP_Dissertation.html"
DOCX_PATH = ROOT / "outputs" / "reports" / "F21MP_Dissertation.docx"
TEMPLATE = ROOT / "MP" / "hwu_cs_dissertation_2025_word_template.docx"


def main() -> int:
    if not HTML_PATH.exists():
        print(
            "Missing HTML. Run: .venv/bin/python3 scripts/build_dissertation_pdf.py",
            file=sys.stderr,
        )
        return 1

    cmd = [
        "pandoc",
        str(HTML_PATH),
        "-o",
        str(DOCX_PATH),
        f"--resource-path={HTML_PATH.parent}",
    ]
    if TEMPLATE.exists():
        cmd.insert(-2, f"--reference-doc={TEMPLATE}")

    subprocess.run(cmd, check=True)
    print(f"Wrote {DOCX_PATH} ({DOCX_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
