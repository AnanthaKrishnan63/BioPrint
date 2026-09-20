"""Experimental login policies. Lower scores are more owner-like.

Background classifiers require exactly matching physical-key feature names.
Synthetic calibration is not evidence of population authentication accuracy.
"""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from contracts import SignalResult
from engine import scorer, supervised

CONFIG = json.loads((Path(__file__).parents[1] / 'experiment.json').read_text())
MODE = CONFIG['mode']


@lru_cache(maxsize=1)
def background():
    path = os.environ.get('BIOPRINT_BACKGROUND')
    if not path:
        return None
    raw = Path(path).read_bytes()
    expected = os.environ.get('BIOPRINT_BACKGROUND_SHA256')
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Typing background checksum missing or mismatched')
    bank = json.loads(raw)
    if bank.get('role') != 'train_background':
        raise RuntimeError('Only TRAIN background data may fit typing models')
    return bank


@lru_cache(maxsize=128)
def fitted(names_json, positives_json):
    bank = background()
    names = json.loads(names_json)
    if bank is None or bank['names'] != names:
        return None
    model = supervised.fit(json.loads(positives_json), bank['fit'], names, c=10., gamma_factor=.1)
    return supervised.calibrate(model, bank['calibration'], target_far=.01)


def typing_signal(conn, user, vector):
    baseline = scorer.Model.from_dict(json.loads(user['keystroke_model']))
    signal = scorer.score(baseline, vector.values, 'keystroke')
    if MODE == 'login-control':
        return signal
    rows = conn.execute("SELECT keystroke_vector FROM samples WHERE user_id=? AND kind='enroll' AND status='ok' ORDER BY id", (user['id'],)).fetchall()
    vectors = [json.loads(r[0]) for r in rows if r[0]][1:]
    values = [r['values'] for r in vectors if r['names'] == vector.names]
    profile = fitted(json.dumps(vector.names), json.dumps(values)) if len(values) >= 10 else None
    if profile is None:
        signal.reasons.insert(0, 'Enrollment-distance profile: no compatible supervised background')
        return signal
    distance = float(supervised.distances(profile, [vector.values])[0])
    # Shift the calibrated signed margin into a nonnegative score with limit 1.
    # This preserves the original acceptance inequality exactly.
    score = max(0., 1. + distance - profile.threshold)
    return SignalResult(name='keystroke', score=score, threshold=1., flagged=score > 1.,
                        reasons=['RBF typing model with a separate TRAIN impostor-calibration threshold'])


def ratio(signal):
    if signal is None or not signal.available or signal.threshold <= 0:
        return math.inf
    value = signal.score / signal.threshold
    return value if math.isfinite(value) else math.inf


def decide(signals, after_step_up=False, keypad_enrolled=False, new_class=False):
    by = {s.name: s for s in signals}
    if by.get('bot') and by['bot'].flagged:
        return 'block', ['Automated or replayed input'] + by['bot'].reasons
    ks = by.get('keystroke')
    r = ratio(ks)
    if after_step_up:
        return ('allow', ['Repeated typing passed the final limit']) if r <= CONFIG['final_ratio'] else ('block', ['Repeated typing did not establish the account holder'])
    if ks and ks.available and r > CONFIG['hard_ratio']:
        return 'block', ['Typing is far outside the enrolled profile']
    dv = by.get('device')
    familiar = bool(dv and dv.available and not dv.flagged)
    if familiar and ks and ks.available and r <= CONFIG['direct_ratio']:
        return 'allow', ['Typing passed the strict direct-login limit on a familiar device']
    if MODE in ('typing-pointer', 'typing-keypad') and keypad_enrolled:
        return 'keypad', ['Additional target checks are needed before granting access']
    if not ks or not ks.available:
        return ('keypad', ['Keyboard timing unavailable; use the enrolled keypad']) if keypad_enrolled else ('retype', ['No usable typing or enrolled keypad profile'])
    return 'step_up', ['Please repeat your typing to resolve uncertain identity evidence']


def decide_keypad(signals, mobile=False):
    by = {s.name: s for s in signals}
    if by.get('bot') and by['bot'].flagged:
        return 'block', ['Automated or replayed target input']
    ks = by.get('keystroke')
    if ks and ks.available and ratio(ks) > CONFIG['hard_ratio']:
        return 'block', ['Target checks cannot override strongly inconsistent typing']
    name = 'keypad_motor' if MODE == 'typing-pointer' and not mobile else 'keypad'
    evidence = by.get(name)
    limit = CONFIG.get('mobile_keypad_ratio', CONFIG['keypad_ratio']) if mobile else CONFIG['keypad_ratio']
    if ratio(evidence) <= limit:
        return 'allow', ['Target movement matched the enrolled profile' if name == 'keypad_motor' else 'Keypad timing matched the enrolled profile']
    return 'block', ['Insufficient or inconsistent target movement' if name == 'keypad_motor' else 'Keypad timing did not establish the account holder']
