"""Enrollment and login step-up using the frozen SapiMouse FCN + cosine model."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import Literal
from contracts import PointerEvent, SignalResult
from engine.pointer_sequence import PointerSequenceEncoder, enroll_sequence_profile

ENCODER_SHA256 = 'a0e613d7029a42f4e25177c75e33b2ecd551a8e1571720df0e9ee0bafb21a32a'
THRESHOLD = 0.9480821490287781  # cosine@5, frozen TRAIN calibration FAR1%
ENROLL_RUNS = 3
ENROLL_MS = 60_000
MIN_POINTS = 641
SCHEMA = '''
CREATE TABLE IF NOT EXISTS neural_pointer_profiles (
 user_id INTEGER PRIMARY KEY REFERENCES users(id), device_class TEXT NOT NULL,
 model_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS neural_pointer_challenges (
 id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), kind TEXT NOT NULL,
 attempt_id INTEGER, expires REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS neural_pointer_runs (
 id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), kind TEXT NOT NULL,
 challenge_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, events_json TEXT NOT NULL,
 embeddings_json TEXT NOT NULL, encoder_sha256 TEXT NOT NULL, signature TEXT NOT NULL);
'''


class ChallengeIn(BaseModel):
    kind: Literal['enroll', 'verify']
    username: str = Field(default='', max_length=64)
    password: str = Field(default='', max_length=128)


class CaptureIn(ChallengeIn):
    challenge_id: str = Field(max_length=128)
    events: list[PointerEvent] = Field(min_length=MIN_POINTS, max_length=20_000)


@lru_cache(maxsize=1)
def encoder():
    path = Path(os.environ.get('BIOPRINT_POINTER_ENCODER', 'missing-pointer-encoder'))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ENCODER_SHA256:
        raise HTTPException(503, 'Trained pointer encoder unavailable or checksum mismatch')
    import torch
    torch.set_num_threads(2)
    return PointerSequenceEncoder(path)


def enrolled(conn, user_id):
    return conn.execute('SELECT * FROM neural_pointer_profiles WHERE user_id=? AND device_class=?',
                        (user_id, 'mouse')).fetchone()


def required(pending):
    return any(s['name'] == 'pointer_neural' for s in json.loads(pending['signals']))


def validate_events(events, kind):
    if any(e.pointer_type != 'mouse' or not e.trusted for e in events):
        raise HTTPException(422, 'Use real mouse movements on this device')
    a = np.asarray([[e.x, e.y, e.t] for e in events], dtype=float)
    if not np.isfinite(a).all() or (np.diff(a[:, 2]) < 0).any() or a[0, 2] < 0:
        raise HTTPException(422, 'Invalid pointer timing or coordinates')
    duration = a[-1, 2] - a[0, 2]
    if duration < (ENROLL_MS if kind == 'enroll' else 3000):
        raise HTTPException(422, 'Keep moving until the recording is complete')
    # Stationary coordinates and button events are part of the trained encoder's
    # input contract. Reject an entirely stationary recording, not valid pauses.
    if not np.any(np.diff(a[:, :2], axis=0)):
        raise HTTPException(422, 'Not enough real movement; please try again')
    return duration


def install(app, db, read_session, pending_step_up, record):
    with db.connect() as conn:
        conn.executescript(SCHEMA)

    def account(conn, body, request):
        if body.kind == 'enroll':
            session = read_session(request)
            if not session:
                raise HTTPException(401, 'Sign in before enrolling pointer movement')
            username = session['username']
        else:
            username = body.username
        user = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if not user or (body.kind == 'verify' and not db.check_password(body.password, user['pw_salt'], user['pw_hash'])):
            raise HTTPException(401, 'Incorrect credentials')
        return user

    def pending_for(conn, user):
        pending = pending_step_up(conn, user['id'], 'keypad')
        if not pending or not required(pending):
            raise HTTPException(409, 'No trained pointer check is pending; sign in again')
        return pending

    @app.get('/api/pointer/status')
    def status(request: Request):
        with db.connect() as conn:
            user = account(conn, ChallengeIn(kind='enroll'), request)
            count = conn.execute("SELECT count(*) FROM neural_pointer_runs WHERE user_id=? AND kind='enroll'",
                                 (user['id'],)).fetchone()[0]
            return {'username': user['username'], 'enrolled': bool(enrolled(conn, user['id'])),
                    'count': count, 'target': ENROLL_RUNS, 'seconds_per_run': ENROLL_MS // 1000,
                    'device_class': 'mouse', 'model': 'SapiMouse FCN + cosine@5'}

    @app.post('/api/pointer/challenge')
    def challenge(body: ChallengeIn, request: Request):
        with db.connect() as conn:
            user = account(conn, body, request)
            if body.kind == 'enroll' and enrolled(conn, user['id']):
                raise HTTPException(409, 'Pointer profile already enrolled; replacement is disabled')
            pending = pending_for(conn, user) if body.kind == 'verify' else None
            if pending and not enrolled(conn, user['id']):
                raise HTTPException(409, 'No trained pointer profile')
            encoder()  # check availability before asking the person to move
            token = secrets.token_urlsafe(24)
            conn.execute('INSERT INTO neural_pointer_challenges (id,user_id,kind,attempt_id,expires) VALUES (?,?,?,?,?)',
                         (token, user['id'], body.kind, pending['id'] if pending else None, time.time() + 180))
            return {'id': token, 'kind': body.kind, 'minimum_points': MIN_POINTS,
                    'minimum_ms': ENROLL_MS if body.kind == 'enroll' else 3000,
                    'expires_in': 180}

    @app.post('/api/pointer/capture')
    def capture(body: CaptureIn, request: Request, response: Response):
        start = time.perf_counter()
        with db.connect() as conn:
            # Serialize consumption and enrollment completion across concurrent posts.
            conn.execute('BEGIN IMMEDIATE')
            user = account(conn, body, request)
            row = conn.execute('SELECT * FROM neural_pointer_challenges WHERE id=?', (body.challenge_id,)).fetchone()
            if not row or row['user_id'] != user['id'] or row['kind'] != body.kind or row['used'] or row['expires'] < time.time():
                raise HTTPException(409, 'This movement recording has expired or was already used')
            pending = pending_for(conn, user) if body.kind == 'verify' else None
            if pending and row['attempt_id'] != pending['id']:
                raise HTTPException(409, 'The login attempt changed; sign in again')
            profile = enrolled(conn, user['id'])
            if body.kind == 'enroll' and profile:
                raise HTTPException(409, 'Pointer profile already enrolled')
            if body.kind == 'verify' and not profile:
                raise HTTPException(409, 'Pointer profile missing')
            duration = validate_events(body.events, body.kind)
            # The browser clock cannot legitimately outrun the issued challenge clock.
            if duration > (time.time() - (row['expires'] - 180)) * 1000 + 2000:
                raise HTTPException(422, 'Recording timing does not match the challenge')
            signature = hashlib.sha256(json.dumps([(e.type, e.x, e.y, round(e.t - body.events[0].t, 3))
                                                   for e in body.events]).encode()).hexdigest()
            if conn.execute('SELECT 1 FROM neural_pointer_runs WHERE signature=?', (signature,)).fetchone():
                raise HTTPException(422, 'This movement recording was already used; record new movement')
            embeddings = encoder().encode(body.events)
            if len(embeddings) < 5 or embeddings.shape[1] != 128 or not np.isfinite(embeddings).all():
                raise HTTPException(422, 'Insufficient usable pointer movement')
            conn.execute('UPDATE neural_pointer_challenges SET used=1 WHERE id=?', (row['id'],))
            conn.execute('INSERT INTO neural_pointer_runs (user_id,kind,challenge_id,created_at,events_json,embeddings_json,encoder_sha256,signature) VALUES (?,?,?,?,?,?,?,?)',
                         (user['id'], body.kind, row['id'], db.now(), json.dumps([e.model_dump() for e in body.events]),
                          json.dumps(embeddings.tolist()), ENCODER_SHA256, signature))
            if body.kind == 'enroll':
                rows = conn.execute("SELECT embeddings_json FROM neural_pointer_runs WHERE user_id=? AND kind='enroll' ORDER BY id", (user['id'],)).fetchall()
                if len(rows) >= ENROLL_RUNS:
                    fitted = enroll_sequence_profile(np.concatenate([json.loads(r[0]) for r in rows]), method='cosine')
                    model = {'center': fitted.model['center'].tolist(), 'encoder_sha256': ENCODER_SHA256,
                             'threshold': THRESHOLD, 'blocks_per_decision': 5, 'enrollment_blocks': fitted.enrollment_blocks}
                    conn.execute('INSERT INTO neural_pointer_profiles VALUES (?,?,?,?)',
                                 (user['id'], 'mouse', json.dumps(model), db.now()))
                return {'accepted': True, 'count': len(rows), 'target': ENROLL_RUNS, 'enrolled': len(rows) >= ENROLL_RUNS}
            model = json.loads(profile['model_json'])
            if model['encoder_sha256'] != ENCODER_SHA256:
                raise HTTPException(409, 'Pointer model version changed; profile migration is required')
            # Use exactly the first complete five-block decision; extra movement
            # is preserved raw, never searched for a favorable five-block window.
            z = embeddings[:5] / np.maximum(np.linalg.norm(embeddings[:5], axis=1, keepdims=True), 1e-12)
            similarity = float(np.clip((z @ np.asarray(model['center'])).mean(), -1, 1))
            signal = SignalResult(name='pointer_neural', score=1 - similarity, threshold=1 - model['threshold'],
                                  flagged=similarity < model['threshold'],
                                  reasons=['Trained SapiMouse FCN with your personal mouse profile; research-calibrated threshold'])
            carried = [SignalResult.model_validate(s) for s in json.loads(pending['signals']) if s['name'] != 'pointer_neural']
            blocked = signal.flagged or any(s.name == 'bot' and s.flagged for s in carried)
            blocked |= any(s.name == 'keystroke' and s.available and s.score > 2 * s.threshold for s in carried)
            return record(conn, response, start, user, None, user['username'], 'block' if blocked else 'allow',
                          ['Pointer movement did not match your profile' if blocked else 'Trained pointer model matched your movement'],
                          carried + [signal])
