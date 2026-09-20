"""BioPrint: behaviour-based login.

Run:  cd code/bioprint && uvicorn server:app --reload
Then: http://localhost:8000
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import re
import secrets
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

import db
from contracts import (KEYPAD_BLANK, KEYPAD_CHALLENGE_TTL_S, KEYPAD_COLS, KEYPAD_DIGITS, KEYPAD_ENROLL_RUNS,
                       KEYPAD_ROWS, KEYPAD_STEPUP_RUNS, KEYPAD_TARGET_LEN, STEP_UP_SAMPLES, STEP_UP_WINDOW_S,
                       AttemptIn, EnrollOut, FeatureVector, KeypadChallenge, KeypadEnrollOut, KeypadIn, KeypadRun,
                       LoginOut, RegisterIn, SignalResult, StepUpIn)
from engine import bot, device, features, keypad, pointer, scorer
from engine.decide import decide, decide_keypad

ENROLL_TARGET = 10  # counted repetitions
# The first repetition is a practice run: stored raw (still useful for replay
# detection) but left out of the profile, because a shaky first attempt widens
# the spread and lets impostors in. On CMU, enrolling while still learning the
# string gave FAR 26% vs 3% when practised.
WARMUP_REPS = 1
MIN_POINTER_SAMPLES = 5  # fit a pointer model only if this many enrollments used the pointer

db.init_db()
app = FastAPI(title="BioPrint")

@app.get("/api/experiment")
def experiment_status():
    return json.loads((Path(__file__).parent / "experiment.json").read_text())


# ---------------------------------------------------------------- session cookie
# An "allow" becomes a real login: a signed, HttpOnly cookie. Stdlib only.
# Value is  base64url(json{"u": username, "since": iso}) + "." + hex(HMAC-SHA256)
# keyed with BIOPRINT_SECRET, or a random per-process key (sessions then die
# with the process, which is what a demo wants). No `Secure` flag: the demo runs
# on plain http://localhost. Only the server can mint one; the browser cannot
# read or forge it.
SESSION_COOKIE = "bioprint_session"
SESSION_MAX_AGE = 12 * 3600  # seconds
_SECRET = (os.environ.get("BIOPRINT_SECRET") or "").encode() or secrets.token_bytes(32)


def _sign(payload: bytes) -> str:
    return hmac.new(_SECRET, payload, hashlib.sha256).hexdigest()


def _session_token(username: str) -> str:
    payload = json.dumps({"u": username, "since": db.now()}, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{body}.{_sign(payload)}"


def read_session(request: Request) -> dict | None:
    """{username, since} if the request carries a valid, unexpired session cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    try:
        payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    try:
        data = json.loads(payload)
        since = datetime.fromisoformat(data["since"])
        username = str(data["u"])
    except (ValueError, KeyError, TypeError):
        return None
    if (datetime.now(timezone.utc) - since).total_seconds() > SESSION_MAX_AGE:
        return None
    return {"username": username, "since": data["since"]}


def start_session(response: Response, username: str) -> None:
    response.set_cookie(SESSION_COOKIE, _session_token(username), max_age=SESSION_MAX_AGE,
                        httponly=True, samesite="lax", path="/")


def end_session(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def _tag_ip(request: Request, *envs: dict) -> None:
    """The client address goes into each env as env["ip"]: a context attribute
    for engine.device, and part of the raw record. Behind the phone's USB
    tunnel both ends are localhost, so it only carries information on a LAN."""
    ip = request.client.host if request.client else None
    for env in envs:
        if isinstance(env, dict):
            env["ip"] = ip


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


def _store_sample(conn, user_id: int, kind: str, sample, kv, pv, status: str = "ok") -> int:
    """Every submission is stored, including the rejected ones: how often a person
    slips and fixes it, or submits a wrong password without noticing, is data."""
    cur = conn.execute(
        "INSERT INTO samples (user_id, kind, created_at, sample_json, keystroke_vector, pointer_vector,"
        " status, corrections) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, kind, db.now(), sample.model_dump_json(), kv.model_dump_json() if kv else None,
         pv.model_dump_json() if pv else None, status, features.correction_count(sample)),
    )
    return cur.lastrowid


def _enroll_rows(conn, user_id: int) -> list[tuple[FeatureVector, FeatureVector | None]]:
    """Every accepted enrollment sample in order: (keystroke vector, pointer vector or None)."""
    rows = conn.execute(
        "SELECT keystroke_vector, pointer_vector FROM samples"
        " WHERE user_id = ? AND kind = 'enroll' AND status = 'ok' ORDER BY id", (user_id,)).fetchall()
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
def enroll(body: AttemptIn, request: Request) -> EnrollOut:
    _tag_ip(request, body.sample.env)
    with db.connect() as conn:
        u = _user(conn, body.username)
        if not u:
            raise HTTPException(404, "unknown user")
        rows = _enroll_rows(conn, u["id"])
        count = max(0, len(rows) - WARMUP_REPS)

        def reject(reason: str, status: str) -> EnrollOut:
            _store_sample(conn, u["id"], "enroll", body.sample, None, None, status)
            return EnrollOut(accepted=False, count=count, target=ENROLL_TARGET,
                             enrolled=u["keystroke_model"] is not None, reasons=[reason])

        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            return reject("wrong password", "wrong_password")
        template = features.template_codes(rows[0][0]) if rows else None
        if why := features.needs_retype(body.sample, template):
            return reject(why, "retype")

        kv = features.keystroke_vector(features.password_keystrokes(body.sample))
        pv = pointer.pointer_vector(body.sample)
        _store_sample(conn, u["id"], "enroll", body.sample, kv, pv)
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


def _record(conn, response: Response, t0: float, user, sample_id: int | None, username: str,
            decision, reasons: list[str], signals: list[SignalResult]) -> LoginOut:
    """Store one attempts row and apply the decision to the session cookie.

    Only an allow signs the browser in. A block also signs it out: whoever is at
    this keyboard just failed the behaviour check, so any session they inherited
    (unlocked laptop, stolen cookie) ends here. Everything else, step_up included,
    leaves the cookie alone.
    """
    latency = (time.perf_counter() - t0) * 1000
    cur = conn.execute(
        "INSERT INTO attempts (user_id, sample_id, username, created_at, decision, reasons, signals, latency_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"] if user else None, sample_id, username, db.now(), decision, json.dumps(reasons),
         json.dumps([s.model_dump() for s in signals]), latency),
    )
    if decision == "allow":
        start_session(response, username)
    elif decision == "block":
        end_session(response)
    return LoginOut(decision=decision, reasons=reasons, signals=signals, latency_ms=latency, attempt_id=cur.lastrowid)


def _enrolled_envs(conn, user_id: int) -> list[dict]:
    return [json.loads(r[0])["env"] for r in conn.execute(
        "SELECT sample_json FROM samples WHERE user_id = ? AND kind = 'enroll' AND status = 'ok'", (user_id,))]


def _pointer_signal(u, pv) -> SignalResult:
    if u["pointer_model"] and pv:
        return scorer.score(scorer.Model.from_dict(json.loads(u["pointer_model"])), pv.values, "pointer")
    return SignalResult(name="pointer", available=False, reasons=["no pointer data to compare"])


@app.post("/api/login")
def login(body: AttemptIn, request: Request, response: Response) -> LoginOut:
    t0 = time.perf_counter()
    _tag_ip(request, body.sample.env)
    signals: list[SignalResult] = []
    sample_id = None

    with db.connect() as conn:
        u = _user(conn, body.username)

        def finish(decision, reasons) -> LoginOut:
            return _record(conn, response, t0, u, sample_id, body.username, decision, reasons, signals)

        if not u:
            return finish("unknown_user", ["no such account"])
        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            sample_id = _store_sample(conn, u["id"], "login", body.sample, None, None, "wrong_password")
            return finish("wrong_password", ["wrong password"])
        if u["keystroke_model"] is None:
            return finish("not_enrolled", ["account has not finished enrollment"])

        ks_model = scorer.Model.from_dict(json.loads(u["keystroke_model"]))
        template = features.template_codes(FeatureVector(names=ks_model.names, values=ks_model.center))
        retype = features.needs_retype(body.sample, template)
        kv = None if retype else features.keystroke_vector(features.password_keystrokes(body.sample))
        pv = pointer.pointer_vector(body.sample)

        sample_id = _store_sample(conn, u["id"], "login", body.sample, kv, pv, "retype" if retype else "ok")

        # Bot check runs even when the sample is unscorable: a script that fumbles
        # the password is still a script.
        prior = [v.values for v in _vectors(conn, u["id"], "keystroke_vector", exclude_id=sample_id)
                 if kv and v.names == kv.names]
        # The device axis needs no keystrokes, so it runs first and every attempt
        # carries it, however the attempt ends (bot, retype, block or allow).
        signals.append(device.check(body.sample.env, _enrolled_envs(conn, u["id"])))
        signals.append(bot.check(body.sample, kv, prior, len(body.password)))
        if signals[-1].flagged:
            return finish(*decide(signals))
        routing = _keypad_routing(u, body.sample.env)
        if retype:
            if features.is_virtual_keyboard(body.sample) and routing["keypad_enrolled"] and signals[0].flagged:
                # A touch keyboard gives no timing, but the keypad works on any
                # device: route there instead of asking for a retype that cannot help.
                signals.append(SignalResult(name="keystroke", available=False, reasons=[retype]))
                return finish(*decide(signals, **routing))
            return finish("retype", [retype])

        signals.append(scorer.score(ks_model, kv.values, "keystroke"))
        signals.append(_pointer_signal(u, pv))

        return finish(*decide(signals, **routing))


def _pending_step_up(conn, user_id: int, decision: str = "step_up"):
    """The attempts row a step-up answers: the user's latest "step_up" (or
    "keypad") decision, recent enough, and not already settled by a later allow
    or block."""
    since = (datetime.now(timezone.utc) - timedelta(seconds=STEP_UP_WINDOW_S)).isoformat(timespec="milliseconds")
    row = conn.execute(
        "SELECT * FROM attempts WHERE user_id = ? AND decision = ? AND created_at >= ?"
        " ORDER BY id DESC LIMIT 1", (user_id, decision, since)).fetchone()
    if not row:
        return None
    settled = conn.execute(
        "SELECT 1 FROM attempts WHERE user_id = ? AND id > ? AND decision IN ('allow', 'block') LIMIT 1",
        (user_id, row["id"])).fetchone()
    return None if settled else row


@app.post("/api/login/stepup")
def login_step_up(body: StepUpIn, request: Request, response: Response) -> LoginOut:
    """Answer to a "step_up" decision: STEP_UP_SAMPLES more typings of the password.

    The pending attempt's keystroke score is read back from attempts.signals; the
    median of it and the new scores is judged against the same threshold. Same
    evidence, more of it: no code, no second device.
    """
    t0 = time.perf_counter()
    _tag_ip(request, *(s.env for s in body.samples))
    signals: list[SignalResult] = []
    sample_id = None

    with db.connect() as conn:
        u = _user(conn, body.username)

        def finish(decision, reasons) -> LoginOut:
            return _record(conn, response, t0, u, sample_id, body.username, decision, reasons, signals)

        if not u:
            return finish("unknown_user", ["no such account"])
        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            for sample in body.samples:
                sample_id = _store_sample(conn, u["id"], "login", sample, None, None, "wrong_password")
            return finish("wrong_password", ["wrong password"])
        pending = _pending_step_up(conn, u["id"])
        if not pending or u["keystroke_model"] is None:
            raise HTTPException(409, "no step-up is pending for this account; sign in first")
        first = next((s for s in json.loads(pending["signals"]) if s["name"] == "keystroke" and s.get("available", True)),
                     None)
        if not first:
            raise HTTPException(409, "the pending attempt has no keystroke score; sign in again")

        ks_model = scorer.Model.from_dict(json.loads(u["keystroke_model"]))
        template = features.template_codes(FeatureVector(names=ks_model.names, values=ks_model.center))

        retypes: list[str] = []
        bots: list[SignalResult] = []
        scored: list[SignalResult] = []
        pvs = []
        for sample in body.samples:
            retype = features.needs_retype(sample, template)
            kv = None if retype else features.keystroke_vector(features.password_keystrokes(sample))
            pv = pointer.pointer_vector(sample)
            sample_id = _store_sample(conn, u["id"], "login", sample, kv, pv, "retype" if retype else "ok")
            prior = [v.values for v in _vectors(conn, u["id"], "keystroke_vector", exclude_id=sample_id)
                     if kv and v.names == kv.names]
            bots.append(bot.check(sample, kv, prior, len(body.password)))
            if retype:
                retypes.append(retype)
            else:
                scored.append(scorer.score(ks_model, kv.values, "keystroke"))
            if pv:
                pvs.append(pv)

        # One bot signal for the batch: the worst of the three.
        signals.append(device.check(body.samples[-1].env, _enrolled_envs(conn, u["id"])))
        signals.append(max(bots, key=lambda b: (b.flagged, b.score)))
        if signals[-1].flagged:
            return finish(*decide(signals))
        if retypes:
            # Simplest honest answer: the whole batch again, straight through.
            return finish("retype", [f"{len(retypes)} of the {len(body.samples)} samples could not be used: "
                                     f"{retypes[0]}; please type all {STEP_UP_SAMPLES} again"])

        scores = [float(first["score"])] + [s.score for s in scored]
        median = float(statistics.median(scores))
        thr = ks_model.threshold
        # Explain with the sample nearest the median: its contributions are the
        # ones the verdict rests on.
        nearest = min(scored, key=lambda s: abs(s.score - median))
        listed = ", ".join(f"{x:.1f}" for x in scores)
        signals.append(SignalResult(
            name="keystroke", score=median, threshold=thr, flagged=median > thr,
            contributions=nearest.contributions,
            reasons=[f"median of {len(scores)} samples ({listed}) vs limit {thr:.1f}"] + nearest.reasons))
        signals.append(_pointer_signal(u, pvs[0] if pvs else None))

        decision, reasons = decide(signals, after_step_up=True, **_keypad_routing(u, body.samples[-1].env))
        headline = f"new device: median of {len(scores)} samples {median:.1f} vs limit {thr:.1f}"
        return finish(decision, [headline] + reasons)


# ---------------------------------------------------------------- scrambled keypad

MOBILE_UA = re.compile(r"Android|iPhone|iPad|Mobile", re.I)


def _env_class(env: dict) -> str:
    """The device class a browser environment implies, before any keypad run
    exists: a phone or tablet is "touch", everything else "mouse". A touch-screen
    laptop used with a mouse counts as "mouse", which is what its keypad runs
    will say too."""
    if not isinstance(env, dict):
        return "unknown"
    if env.get("ua_mobile") is True or MOBILE_UA.search(str(env.get("ua", ""))):
        return "touch"
    return "mouse"


def _keypad_routing(u, env: dict) -> dict:
    """Keyword arguments for decide(): can the keypad be asked for, and is this a
    different kind of device from the one it was enrolled on?"""
    if not u["keypad_model"]:
        return {"keypad_enrolled": False, "new_class": False}
    enrolled = json.loads(u["keypad_model"]).get("device_class", "unknown")
    now = _env_class(env)
    return {"keypad_enrolled": True, "new_class": enrolled != "unknown" and now != "unknown" and now != enrolled}


def _new_challenge(conn, username: str) -> KeypadChallenge:
    cells = KEYPAD_COLS * KEYPAD_ROWS
    layout = list(range(KEYPAD_DIGITS)) + [KEYPAD_BLANK] * (cells - KEYPAD_DIGITS)
    random.shuffle(layout)
    target: list[int] = []
    while len(target) < KEYPAD_TARGET_LEN:
        d = random.randrange(KEYPAD_DIGITS)
        if len(target) >= 2 and target[-1] == d and target[-2] == d:
            continue  # three in a row is a dull captcha and a degenerate sample
        target.append(d)
    now = datetime.now(timezone.utc)
    ch = KeypadChallenge(id=secrets.token_urlsafe(12), layout=layout, target=target,
                         expires_at=(now + timedelta(seconds=KEYPAD_CHALLENGE_TTL_S)).isoformat(timespec="milliseconds"))
    conn.execute("INSERT INTO keypad_challenges (id, username, layout, target, created_at, expires_at)"
                 " VALUES (?, ?, ?, ?, ?, ?)",
                 (ch.id, username, json.dumps(layout), json.dumps(target), db.now(), ch.expires_at))
    conn.execute("DELETE FROM keypad_challenges WHERE expires_at < ?",
                 ((now - timedelta(days=1)).isoformat(timespec="milliseconds"),))
    return ch


def _check_run(conn, username: str, run: KeypadRun) -> str | None:
    """Why this run cannot be used, or None. Consumes the challenge."""
    row = conn.execute("SELECT * FROM keypad_challenges WHERE id = ?", (run.challenge_id,)).fetchone()
    if not row or row["username"] != username:
        return "unknown keypad challenge"
    if row["used"]:
        return "this keypad challenge was already answered"
    if row["expires_at"] < db.now():
        return "this keypad challenge expired; solve a fresh one"
    conn.execute("UPDATE keypad_challenges SET used = 1 WHERE id = ?", (run.challenge_id,))
    if run.layout != json.loads(row["layout"]) or run.target != json.loads(row["target"]):
        return "the keypad layout or target does not match the one issued"
    if not run.completed or keypad.entered_digits(run) != run.target:
        return "the digits entered did not match the target"
    return None


def _store_run(conn, user_id: int, kind: str, run: KeypadRun, status: str = "ok",
               attempt_id: int | None = None) -> tuple[int, list[FeatureVector], list[FeatureVector]]:
    cog, motor = keypad.tap_vectors(run) if status == "ok" else ([], [])
    cur = conn.execute(
        "INSERT INTO keypad_runs (user_id, kind, attempt_id, created_at, run_json, cog_vectors, motor_vectors,"
        " device_class, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, kind, attempt_id, db.now(), run.model_dump_json(),
         json.dumps([v.model_dump() for v in cog]), json.dumps([v.model_dump() for v in motor]),
         keypad.device_class(run), status))
    return cur.lastrowid, cog, motor


def _enroll_runs(conn, user_id: int) -> list[KeypadRun]:
    return [KeypadRun.model_validate_json(r[0]) for r in conn.execute(
        "SELECT run_json FROM keypad_runs WHERE user_id = ? AND kind = 'enroll' AND status = 'ok' ORDER BY id",
        (user_id,))]


@app.post("/api/keypad/challenge")
def keypad_challenge(body: dict) -> KeypadChallenge:
    username = str(body.get("username", ""))[:64]
    with db.connect() as conn:
        if not username or not _user(conn, username):
            raise HTTPException(404, "unknown user")
        return _new_challenge(conn, username)


@app.post("/api/enroll/keypad")
def enroll_keypad(body: KeypadIn, request: Request) -> KeypadEnrollOut:
    """One solved captcha per call. After KEYPAD_ENROLL_RUNS accepted runs the
    keypad profile is fitted and the account can be stepped up on a new device."""
    if len(body.runs) != 1:
        raise HTTPException(400, "enroll one keypad run per call")
    run = body.runs[0]
    _tag_ip(request, run.env)
    with db.connect() as conn:
        u = _user(conn, body.username)
        if not u:
            raise HTTPException(404, "unknown user")
        runs = _enroll_runs(conn, u["id"])
        enrolled = u["keypad_model"] is not None

        def reject(reason: str, status: str) -> KeypadEnrollOut:
            _store_run(conn, u["id"], "enroll", run, status)
            return KeypadEnrollOut(accepted=False, count=len(runs), target=KEYPAD_ENROLL_RUNS, enrolled=enrolled,
                                   device_class=keypad.device_class(run), reasons=[reason])

        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            return reject("wrong password", "wrong_password")
        if why := _check_run(conn, body.username, run):
            return reject(why, "bad_challenge" if "challenge" in why or "issued" in why else "incomplete")
        _, cog, _ = _store_run(conn, u["id"], "enroll", run)
        if not cog:
            return reject("no usable taps in this run", "incomplete")
        runs.append(run)
        reasons: list[str] = []
        if len(runs) >= KEYPAD_ENROLL_RUNS and not enrolled:
            profile = keypad.fit_profile(runs)
            conn.execute("UPDATE users SET keypad_model = ? WHERE id = ?", (json.dumps(profile.to_dict()), u["id"]))
            enrolled = True
            reasons.append(f"keypad profile built from {len(runs)} runs on a {profile.device_class} device")
    return KeypadEnrollOut(accepted=True, count=len(runs), target=KEYPAD_ENROLL_RUNS, enrolled=enrolled,
                           device_class=keypad.device_class(run), reasons=reasons)


@app.post("/api/login/keypad")
def login_keypad(body: KeypadIn, request: Request, response: Response) -> LoginOut:
    """Answer to a "keypad" decision: KEYPAD_STEPUP_RUNS solved captchas. The
    keypad's cognitive score decides; movement is advisory; a bot flag blocks."""
    t0 = time.perf_counter()
    _tag_ip(request, *(r.env for r in body.runs))
    signals: list[SignalResult] = []

    with db.connect() as conn:
        u = _user(conn, body.username)

        def finish(decision, reasons, run_ids: list[int] = ()) -> LoginOut:
            out = _record(conn, response, t0, u, None, body.username, decision, reasons, signals)
            for rid in run_ids:
                conn.execute("UPDATE keypad_runs SET attempt_id = ? WHERE id = ?", (out.attempt_id, rid))
            return out

        if not u:
            return finish("unknown_user", ["no such account"])
        if not db.check_password(body.password, u["pw_salt"], u["pw_hash"]):
            ids = [_store_run(conn, u["id"], "stepup", r, "wrong_password")[0] for r in body.runs]
            return finish("wrong_password", ["wrong password"], ids)
        pending = _pending_step_up(conn, u["id"], "keypad")
        if not pending or u["keypad_model"] is None:
            raise HTTPException(409, "no keypad check is pending for this account; sign in first")
        if len(body.runs) != KEYPAD_STEPUP_RUNS:
            raise HTTPException(400, f"{KEYPAD_STEPUP_RUNS} keypad runs are needed")

        # The pending attempt's own signals (device, keystroke) travel with the verdict.
        carried = [SignalResult.model_validate(s) for s in json.loads(pending["signals"])
                   if s["name"] in ("keystroke",)]
        signals.append(device.check(body.runs[-1].env, _enrolled_envs(conn, u["id"])))
        signals.append(keypad.bot_check(body.runs))
        if signals[-1].flagged:
            ids = [_store_run(conn, u["id"], "stepup", r)[0] for r in body.runs]
            return finish(*decide_keypad(signals), ids)

        bad = [why for r in body.runs if (why := _check_run(conn, body.username, r))]
        if bad:
            ids = [_store_run(conn, u["id"], "stepup", r, "incomplete")[0] for r in body.runs]
            return finish("retype", [f"the keypad checks could not be used: {bad[0]}; please solve them again"], ids)
        ids = [_store_run(conn, u["id"], "stepup", r)[0] for r in body.runs]

        profile = keypad.Profile.from_dict(json.loads(u["keypad_model"]))
        cog, motor = keypad.score_runs(profile, body.runs)
        signals += [cog, motor] + carried
        return finish(*decide_keypad(signals), ids)


@app.get("/api/session")
def session(request: Request) -> dict:
    s = read_session(request)
    if not s:
        raise HTTPException(401, "not signed in")
    return s


@app.post("/api/logout")
def logout(response: Response) -> dict:
    end_session(response)
    return {"ok": True}


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
        n = conn.execute("SELECT COUNT(*) FROM samples WHERE user_id = ? AND kind = 'enroll' AND status = 'ok'",
                         (u["id"],)).fetchone()[0]
        # Correction habits across every submission, accepted or not.
        h = conn.execute(
            """SELECT COUNT(*) AS submissions,
                      SUM(status = 'retype') AS corrected,
                      SUM(status = 'wrong_password') AS wrong_password,
                      COALESCE(AVG(corrections), 0) AS mean_corrections
               FROM samples WHERE user_id = ?""", (u["id"],)).fetchone()
        habits = {
            "submissions": h["submissions"],
            "corrected": h["corrected"] or 0,               # slipped and fixed it
            "wrong_password": h["wrong_password"] or 0,     # slipped and did not notice
            "correction_rate": (h["corrected"] or 0) / h["submissions"] if h["submissions"] else 0.0,
            "wrong_password_rate": (h["wrong_password"] or 0) / h["submissions"] if h["submissions"] else 0.0,
            "mean_corrections": h["mean_corrections"],
        }
        kp_n = conn.execute("SELECT COUNT(*) FROM keypad_runs WHERE user_id = ? AND kind = 'enroll' AND status = 'ok'",
                            (u["id"],)).fetchone()[0]
        kp_class = json.loads(u["keypad_model"])["device_class"] if u["keypad_model"] else "unknown"
        kp_runs = [KeypadRun.model_validate_json(r[0]) for r in conn.execute(
            "SELECT run_json FROM keypad_runs WHERE user_id = ? AND kind = 'enroll'", (u["id"],))]
    kp_stats = [keypad.run_stats(r) for r in kp_runs]
    habits["keypad_runs"] = len(kp_stats)
    habits["keypad_errors"] = sum(st.get("errors", 0) for st in kp_stats)
    habits["keypad_mean_duration_ms"] = (sum(st.get("duration_ms", 0.0) for st in kp_stats) / len(kp_stats)
                                         if kp_stats else 0.0)
    return {"username": username, "enroll_count": max(0, n - WARMUP_REPS), "enroll_target": ENROLL_TARGET,
            "warmup_reps": WARMUP_REPS, "pointer_enrolled": u["pointer_model"] is not None, "habits": habits,
            "enrolled": u["keystroke_model"] is not None,
            "keypad_enrolled": u["keypad_model"] is not None, "keypad_count": kp_n,
            "keypad_target": KEYPAD_ENROLL_RUNS, "keypad_device_class": kp_class}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
