"""BioPrint: behaviour-based login.

Run:  cd code/bioprint && uvicorn server:app --reload
Then: http://localhost:8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

import db
from contracts import AttemptIn, EnrollOut, FeatureVector, LoginOut, RegisterIn, SignalResult
from engine import bot, features, pointer, scorer
from engine.decide import decide

ENROLL_TARGET = 10  # counted repetitions
# The first repetition is a practice run: stored raw (still useful for replay
# detection) but left out of the profile, because a shaky first attempt widens
# the spread and lets impostors in. On CMU, enrolling while still learning the
# string gave FAR 26% vs 3% when practised.
WARMUP_REPS = 1
MIN_POINTER_SAMPLES = 5  # fit a pointer model only if this many enrollments used the pointer

db.init_db()
app = FastAPI(title="BioPrint")


def _user(conn, username: str):
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def _vectors(conn, user_id: int, column: str, kind: str | None = None,
             exclude_id: int | None = None) -> list[FeatureVector]:
    q = f"SELECT {column} FROM samples WHERE user_id = ? AND {column} IS NOT NULL AND id IS NOT ?"
    args: list = [user_id, exclude_id]
    if kind:
        q += " AND kind = ?"
        args.append(kind)
    q += " ORDER BY id"
    return [FeatureVector.model_validate_json(r[0]) for r in conn.execute(q, args)]


def _enroll_rows(conn, user_id: int) -> list[tuple[FeatureVector, FeatureVector | None]]:
    """Every accepted enrollment sample in order: (keystroke vector, pointer vector or None)."""
    rows = conn.execute(
        "SELECT keystroke_vector, pointer_vector FROM samples"
        " WHERE user_id = ? AND kind = 'enroll' ORDER BY id", (user_id,)).fetchall()
    return [(FeatureVector.model_validate_json(k), FeatureVector.model_validate_json(p) if p else None)
            for k, p in rows]


def _fit(vectors: list[FeatureVector]) -> scorer.Model:
    return scorer.fit([v.values for v in vectors], vectors[0].names)


@app.post("/api/register")
def register(body: RegisterIn) -> dict:
    salt, h = db.hash_password(body.password)
    with db.connect() as conn:
        if _user(conn, body.username):
            raise HTTPException(409, "username taken")
        cur = conn.execute(
            "INSERT INTO users (username, pw_salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
            (body.username, salt, h, db.now()),
        )
    return {"id": cur.lastrowid, "enroll_target": ENROLL_TARGET}


@app.post("/api/enroll")
def enroll(body: AttemptIn) -> EnrollOut:
    with db.connect() as conn:
        u = _user(conn, body.username)
        if not u:
            raise HTTPException(404, "unknown user")
        rows = _enroll_rows(conn, u["id"])
        count = max(0, len(rows) - WARMUP_REPS)

        def reject(reason: str) -> EnrollOut:
            return EnrollOut(accepted=False, count=count, target=ENROLL_TARGET,
                             enrolled=u["keystroke_model"] is not None, reasons=[reason])

        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            return reject("wrong password")
        template = features.template_codes(rows[0][0]) if rows else None
        if why := features.needs_retype(body.sample, template):
            return reject(why)

        kv = features.keystroke_vector(features.password_keystrokes(body.sample))
        pv = pointer.pointer_vector(body.sample)
        conn.execute(
            "INSERT INTO samples (user_id, kind, created_at, sample_json, keystroke_vector, pointer_vector)"
            " VALUES (?, 'enroll', ?, ?, ?, ?)",
            (u["id"], db.now(), body.sample.model_dump_json(), kv.model_dump_json(),
             pv.model_dump_json() if pv else None),
        )
        rows.append((kv, pv))
        warmup = len(rows) <= WARMUP_REPS
        count = len(rows) - WARMUP_REPS
        reasons = ["practice run, not counted"] if warmup else []

        enrolled = u["keystroke_model"] is not None
        if count >= ENROLL_TARGET:
            counted = rows[WARMUP_REPS:]
            ks_model = _fit([k for k, _ in counted])
            pvs = [p for _, p in counted if p]
            pt_model = _fit(pvs) if len(pvs) >= MIN_POINTER_SAMPLES else None
            conn.execute(
                "UPDATE users SET keystroke_model = ?, pointer_model = ? WHERE id = ?",
                (json.dumps(ks_model.to_dict()), json.dumps(pt_model.to_dict()) if pt_model else None, u["id"]),
            )
            enrolled = True
    return EnrollOut(accepted=True, count=max(0, count), target=ENROLL_TARGET, enrolled=enrolled,
                     warmup=warmup, reasons=reasons)


@app.post("/api/login")
def login(body: AttemptIn) -> LoginOut:
    t0 = time.perf_counter()
    signals: list[SignalResult] = []
    sample_id = None

    with db.connect() as conn:
        u = _user(conn, body.username)

        def finish(decision, reasons) -> LoginOut:
            latency = (time.perf_counter() - t0) * 1000
            cur = conn.execute(
                "INSERT INTO attempts (user_id, sample_id, username, created_at, decision, reasons, signals, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (u["id"] if u else None, sample_id, body.username, db.now(), decision, json.dumps(reasons),
                 json.dumps([s.model_dump() for s in signals]), latency),
            )
            return LoginOut(decision=decision, reasons=reasons, signals=signals,
                            latency_ms=latency, attempt_id=cur.lastrowid)

        if not u:
            return finish("unknown_user", ["no such account"])
        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            return finish("wrong_password", ["wrong password"])
        if u["keystroke_model"] is None:
            return finish("not_enrolled", ["account has not finished enrollment"])

        ks_model = scorer.Model.from_dict(json.loads(u["keystroke_model"]))
        template = features.template_codes(FeatureVector(names=ks_model.names, values=ks_model.center))
        retype = features.needs_retype(body.sample, template)
        kv = None if retype else features.keystroke_vector(features.password_keystrokes(body.sample))
        pv = pointer.pointer_vector(body.sample)

        cur = conn.execute(
            "INSERT INTO samples (user_id, kind, created_at, sample_json, keystroke_vector, pointer_vector)"
            " VALUES (?, 'login', ?, ?, ?, ?)",
            (u["id"], db.now(), body.sample.model_dump_json(), kv.model_dump_json() if kv else None,
             pv.model_dump_json() if pv else None),
        )
        sample_id = cur.lastrowid

        # Bot check runs even when the sample is unscorable: a script that fumbles
        # the password is still a script.
        prior = [v.values for v in _vectors(conn, u["id"], "keystroke_vector", exclude_id=sample_id)
                 if kv and v.names == kv.names]
        signals.append(bot.check(body.sample, kv, prior, len(body.password)))
        if signals[-1].flagged:
            return finish(*decide(signals))
        if retype:
            return finish("retype", [retype])

        signals.append(scorer.score(ks_model, kv.values, "keystroke"))
        if u["pointer_model"] and pv:
            signals.append(scorer.score(scorer.Model.from_dict(json.loads(u["pointer_model"])), pv.values, "pointer"))
        else:
            signals.append(SignalResult(name="pointer", available=False, reasons=["no pointer data to compare"]))

        return finish(*decide(signals))


@app.get("/api/attempts")
def attempts(user: str | None = None, limit: int = 50) -> list[dict]:
    q = "SELECT * FROM attempts"
    args: list = []
    if user:
        q += " WHERE username = ?"
        args.append(user)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(min(limit, 500))
    with db.connect() as conn:
        rows = conn.execute(q, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["reasons"] = json.loads(d["reasons"])
        d["signals"] = json.loads(d["signals"])
        out.append(d)
    return out


@app.post("/api/attempts/{attempt_id}/label")
def label_attempt(attempt_id: int, body: dict) -> dict:
    """Ground truth for evaluation: genuine | impostor | bot | null to clear."""
    label = body.get("label")
    if label not in ("genuine", "impostor", "bot", None):
        raise HTTPException(400, "label must be genuine, impostor, bot or null")
    with db.connect() as conn:
        if not conn.execute("UPDATE attempts SET label = ? WHERE id = ?", (label, attempt_id)).rowcount:
            raise HTTPException(404, "unknown attempt")
    return {"id": attempt_id, "label": label}


@app.get("/api/users/{username}")
def user_status(username: str) -> dict:
    with db.connect() as conn:
        u = _user(conn, username)
        if not u:
            raise HTTPException(404, "unknown user")
        n = conn.execute("SELECT COUNT(*) FROM samples WHERE user_id = ? AND kind = 'enroll'", (u["id"],)).fetchone()[0]
    return {"username": username, "enroll_count": max(0, n - WARMUP_REPS), "enroll_target": ENROLL_TARGET,
            "warmup_reps": WARMUP_REPS,
            "enrolled": u["keystroke_model"] is not None}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
