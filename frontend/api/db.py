"""SQLite store for moderator queue and audit trail (frontend-local only)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "moderator_cases.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                message_preview TEXT NOT NULL,
                model_id TEXT NOT NULL,
                probability REAL NOT NULL,
                predicted INTEGER NOT NULL,
                tier INTEGER NOT NULL,
                tier_reason TEXT NOT NULL,
                peerness REAL,
                combined_score REAL,
                is_repeat_offender INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewer_note TEXT,
                reviewed_at TEXT
            );
            """
        )


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_case(
    message_preview: str,
    model_id: str,
    probability: float,
    predicted: bool,
    tier: int,
    tier_reason: str,
    peerness: float | None,
    combined_score: float | None,
    is_repeat_offender: int | None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO cases (
                created_at, message_preview, model_id, probability, predicted,
                tier, tier_reason, peerness, combined_score, is_repeat_offender, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                _utc_now(),
                message_preview[:500],
                model_id,
                probability,
                int(predicted),
                tier,
                tier_reason,
                peerness,
                combined_score,
                is_repeat_offender,
            ),
        )
        return int(cur.lastrowid)


def list_queue(status: str | None = "pending"):
    with connect() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM cases WHERE status = ?
                ORDER BY tier ASC, combined_score DESC, created_at DESC
                """,
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cases ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]


def review_case(case_id: int, decision: str, note: str = "") -> bool:
    if decision not in {"approved", "dismissed", "escalated"}:
        raise ValueError("decision must be approved, dismissed, or escalated")
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE cases SET status = ?, reviewer_note = ?, reviewed_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (decision, note, _utc_now(), case_id),
        )
        return cur.rowcount > 0
