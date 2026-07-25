#!/usr/bin/env python3
"""Capture full-page moderator dashboard screenshots for the dissertation."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FULL = ROOT / "outputs" / "media" / "frontend_dashboard_full.png"
OUT_RESULT = ROOT / "outputs" / "media" / "frontend_dashboard_result.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE = "http://127.0.0.1:8080"
DEMO_MESSAGE = (
    "If you continue this behaviour I am going to report you and "
    "get you blocked from the site."
)


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(retries: int = 20) -> None:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError("Dashboard server not reachable at http://127.0.0.1:8080")


def chrome_screenshot(
    url: str,
    out: Path,
    width: int = 1280,
    height: int = 1100,
    virtual_time_budget_ms: int = 0,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
    ]
    if virtual_time_budget_ms > 0:
        cmd.append(f"--virtual-time-budget={virtual_time_budget_ms}")
    cmd.extend([f"--screenshot={out}", url])
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    wait_for_server()
    post_json(
        "/api/predict",
        {
            "message": DEMO_MESSAGE,
            "relational": {
                "peerness": 0.93,
                "total_msgs": 2,
                "aggressive_msgs": 1,
            },
            "prefer_full_model": True,
            "add_to_queue": True,
        },
    )
    chrome_screenshot(BASE, OUT_FULL)
    chrome_screenshot(
        f"{BASE}/?capture=result",
        OUT_RESULT,
        width=900,
        height=1250,
        virtual_time_budget_ms=4000,
    )
    print(f"Wrote {OUT_FULL}")
    print(f"Wrote {OUT_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
