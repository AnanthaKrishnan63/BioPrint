"""SQLite storage. Raw samples are the record of truth; vectors and models are
derived and can be recomputed from `samples.sample_json`.

HACKATHON ONLY: samples keep event.code for the password field, so the password
is recoverable from this database. Never reuse this schema for real users.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB = Path(os.environ.get("BIOPRINT_DB", Path(__file__).parent / "bioprint.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL UNIQUE,
    pw_salt         BLOB    NOT NULL,
    pw_hash         BLOB    NOT NULL,
    created_at      TEXT    NOT NULL,
    keystroke_model TEXT,               -- JSON from scorer.Model.to_dict(); NULL until enrolled
    pointer_model   TEXT,               -- JSON; NULL if too few enrollment samples had pointer data
    keypad_model    TEXT                -- JSON from engine.keypad.Profile.to_dict(); NULL until the keypad runs are done
);

CREATE TABLE IF NOT EXISTS samples (    -- RAW. One per enrollment repetition or login attempt.
    id               INTEGER PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind             TEXT    NOT NULL CHECK (kind IN ('enroll', 'login')),
    created_at       TEXT    NOT NULL,
    sample_json      TEXT    NOT NULL,  -- contracts.Sample, verbatim
    keystroke_vector TEXT,              -- JSON FeatureVector, derived; NULL when status != 'ok'
    pointer_vector   TEXT,              -- JSON FeatureVector, derived
    status           TEXT    NOT NULL DEFAULT 'ok',  -- ok | retype | wrong_password
    corrections      INTEGER            -- Backspace/Delete/arrow presses in the password field
);

CREATE TABLE IF NOT EXISTS keypad_challenges (  -- issued layouts; a run must answer one of these
    id          TEXT    PRIMARY KEY,
    username    TEXT    NOT NULL,
    layout      TEXT    NOT NULL,      -- JSON list[int]
    target      TEXT    NOT NULL,      -- JSON list[int]
    created_at  TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS keypad_runs (    -- RAW. One per scrambled-keypad captcha, enrollment or step-up.
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind          TEXT    NOT NULL CHECK (kind IN ('enroll', 'stepup')),
    attempt_id    INTEGER REFERENCES attempts(id) ON DELETE SET NULL,
    created_at    TEXT    NOT NULL,
    run_json      TEXT    NOT NULL,    -- contracts.KeypadRun, verbatim
    cog_vectors   TEXT,                -- JSON list[FeatureVector], one per tap, derived
    motor_vectors TEXT,                -- JSON list[FeatureVector], one per tap, derived
    device_class  TEXT    NOT NULL DEFAULT 'unknown',
    status        TEXT    NOT NULL DEFAULT 'ok'  -- ok | incomplete | wrong_password | bad_challenge
);

CREATE TABLE IF NOT EXISTS attempts (   -- one per /api/login call, including failures
    id           INTEGER PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    sample_id    INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    username     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    decision     TEXT    NOT NULL,
    reasons      TEXT    NOT NULL,      -- JSON list[str]
    signals      TEXT    NOT NULL,      -- JSON list[SignalResult]
    latency_ms   REAL    NOT NULL,
    label        TEXT                   -- optional ground truth for evaluation: genuine | impostor | bot
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


# Columns added after the first databases existed. CREATE TABLE IF NOT EXISTS
# never alters an existing table, so each is added here if missing.
MIGRATIONS = [
    ("samples", "status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("samples", "corrections", "INTEGER"),
    ("users", "keypad_model", "TEXT"),
]


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        for table, column, decl in MIGRATIONS:
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# scrypt from the standard library: no extra dependency, and a slow hash is the
# right default even for a demo.
def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or os.urandom(16)
    return salt, hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


def check_password(password: str, salt: bytes, expected: bytes) -> bool:
    return hmac.compare_digest(hash_password(password, salt)[1], expected)
