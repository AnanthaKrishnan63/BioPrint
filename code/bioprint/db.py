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
    pointer_model   TEXT                -- JSON; NULL if too few enrollment samples had pointer data
);

CREATE TABLE IF NOT EXISTS samples (    -- RAW. One per enrollment repetition or login attempt.
    id               INTEGER PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind             TEXT    NOT NULL CHECK (kind IN ('enroll', 'login')),
    created_at       TEXT    NOT NULL,
    sample_json      TEXT    NOT NULL,  -- contracts.Sample, verbatim
    keystroke_vector TEXT,              -- JSON FeatureVector, derived
    pointer_vector   TEXT               -- JSON FeatureVector, derived
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


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# scrypt from the standard library: no extra dependency, and a slow hash is the
# right default even for a demo.
def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or os.urandom(16)
    return salt, hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


def check_password(password: str, salt: bytes, expected: bytes) -> bool:
    return hmac.compare_digest(hash_password(password, salt)[1], expected)
