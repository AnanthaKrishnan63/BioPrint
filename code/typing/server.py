"""Local collector for typing sessions.

Serves the page and accepts finished sessions into a SQLite file.

Run:  conda run -n bigidea uvicorn server:app --reload
Then: http://localhost:8000
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
DB = BASE / "sessions.db"

# Raw events and derived metrics live in separate tables on purpose. The events
# are the record of truth; the metrics are a convenience copy of what the page
# displayed. Feature extraction will change as the research moves on, and when it
# does we recompute from `keystrokes` rather than re-collecting from people.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT    NOT NULL,          -- ISO 8601, UTC, server clock
    subject      TEXT,                      -- free-text label, may be null
    user_agent   TEXT    NOT NULL,
    screen       TEXT    NOT NULL,          -- "1920x1080@2"
    timer_res_ms REAL,                      -- observed clock granularity, may be null
    had_paste    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS keystrokes (     -- RAW. Never overwritten, never derived.
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    code       TEXT    NOT NULL,            -- event.code only; never the character
    type       TEXT    NOT NULL CHECK (type IN ('down', 'up')),
    t_ms       REAL    NOT NULL,            -- ms since the session's first event
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS session_metrics (-- DERIVED. Recomputable from keystrokes.
    session_id       INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    duration_ms      REAL,
    total_keys       INTEGER,
    text_keys        INTEGER,
    keystrokes_gross INTEGER,
    net_chars        INTEGER,
    gross_wpm        REAL,
    net_wpm          REAL,
    correction_ratio REAL,
    backspace_count  INTEGER,
    delete_count     INTEGER,
    mean_dwell_ms    REAL,
    mean_flight_ms   REAL,
    negative_flights INTEGER
);
"""

MAX_EVENTS = 200_000  # a very long session is ~20k; this is a sanity bound


class KeyEvent(BaseModel):
    code: str = Field(max_length=32)
    type: Literal["down", "up"]
    t: float


class SessionIn(BaseModel):
    subject: str | None = Field(default=None, max_length=64)
    user_agent: str = Field(max_length=512)
    screen: str = Field(max_length=64)
    timer_res_ms: float | None = None
    had_paste: bool = False
    net_chars: int = 0
    metrics: dict = Field(default_factory=dict)
    events: list[KeyEvent] = Field(max_length=MAX_EVENTS)


@contextmanager
def connect():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


init_db()
app = FastAPI(title="Typing capture")


# Only the metric names the table knows about are stored, so a change on the page
# cannot silently write junk columns or crash the insert.
METRIC_COLUMNS = {
    "durationMs": "duration_ms",
    "totalKeys": "total_keys",
    "textKeys": "text_keys",
    "keystrokesGross": "keystrokes_gross",
    "netChars": "net_chars",
    "grossWpm": "gross_wpm",
    "netWpm": "net_wpm",
    "correctionRatio": "correction_ratio",
    "backspaceCount": "backspace_count",
    "deleteCount": "delete_count",
    "meanDwellMs": "mean_dwell_ms",
    "meanFlightMs": "mean_flight_ms",
    "negativeFlights": "negative_flights",
}


@app.post("/api/session")
def save_session(payload: SessionIn) -> dict:
    if not payload.events:
        raise HTTPException(status_code=400, detail="session has no key events")

    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (started_at, subject, user_agent, screen, timer_res_ms, had_paste)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                payload.subject,
                payload.user_agent,
                payload.screen,
                payload.timer_res_ms,
                int(payload.had_paste),
            ),
        )
        session_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO keystrokes (session_id, seq, code, type, t_ms) VALUES (?, ?, ?, ?, ?)",
            [
                (session_id, i, e.code, e.type, e.t)
                for i, e in enumerate(payload.events)
            ],
        )

        cols = ["session_id"]
        vals = [session_id]
        for js_name, col in METRIC_COLUMNS.items():
            if js_name in payload.metrics:
                cols.append(col)
                vals.append(payload.metrics[js_name])
        placeholders = ", ".join("?" * len(cols))
        conn.execute(
            f"INSERT INTO session_metrics ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )

    return {"id": session_id, "keystrokes": len(payload.events)}


@app.get("/api/sessions")
def list_sessions(limit: int = 50) -> list[dict]:
    """Recent sessions with their metrics, newest first. For eyeballing the data."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT s.*, m.*, (SELECT COUNT(*) FROM keystrokes k
                                 WHERE k.session_id = s.id) AS event_count
               FROM sessions s LEFT JOIN session_metrics m ON m.session_id = s.id
               ORDER BY s.id DESC LIMIT ?""",
            (min(limit, 500),),
        ).fetchall()
    return [dict(r) for r in rows]


# Mounted last so the /api routes above take precedence.
app.mount("/", StaticFiles(directory=BASE, html=True), name="static")
