import json
from types import SimpleNamespace
import numpy as np
import pytest


def setup(client, monkeypatch, username='pointer-user'):
    import pointer_neural as neural
    import server
    assert client.post('/api/register', json={'username': username, 'password': 'test-only'}).status_code == 200
    client.cookies.set(server.SESSION_COOKIE, server._session_token(username))
    fake = SimpleNamespace(encode=lambda events: np.tile(np.r_[1., np.zeros(127)], (len(events) // 128, 1)))
    monkeypatch.setattr(neural, 'encoder', lambda: fake)
    clock = [1000.]
    monkeypatch.setattr(neural.time, 'time', lambda: clock[0])
    return fake, clock


def events(seed=0, seconds=60):
    return [{'type': 'move', 't': i * seconds * 1000 / 640, 'x': (i + seed) % 400,
             'y': (i * 3 + seed) % 250, 'pointer_type': 'mouse', 'trusted': True}
            for i in range(641)]


def enroll(client, clock):
    for i in range(3):
        challenge = client.post('/api/pointer/challenge', json={'kind': 'enroll'})
        assert challenge.status_code == 200, challenge.text
        clock[0] += 61
        reply = client.post('/api/pointer/capture', json={'kind': 'enroll', 'challenge_id': challenge.json()['id'], 'events': events(i)})
        assert reply.status_code == 200, reply.text
        assert reply.json()['enrolled'] is (i == 2)


def pending(username='pointer-user'):
    import db
    with db.connect() as conn:
        user = conn.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()[0]
        signals = [dict(name='pointer_neural', available=False), dict(name='keystroke', score=1., threshold=1.)]
        conn.execute('INSERT INTO attempts (user_id,username,created_at,decision,reasons,signals,latency_ms) VALUES (?,?,?,?,?,?,0)',
                     (user, username, db.now(), 'keypad', '[]', json.dumps(signals)))


def test_enrollment_requires_signed_in_account(client, monkeypatch):
    setup(client, monkeypatch)
    client.cookies.clear()
    assert client.post('/api/pointer/challenge', json={'kind': 'enroll', 'username': 'pointer-user', 'password': 'test-only'}).status_code == 401


def test_enrollment_is_per_user_and_preserves_raw_data(client, monkeypatch):
    _, clock = setup(client, monkeypatch)
    enroll(client, clock)
    assert client.get('/api/pointer/status').json()['enrolled']
    assert client.post('/api/pointer/challenge', json={'kind': 'enroll'}).status_code == 409
    import db
    with db.connect() as conn:
        rows = conn.execute('SELECT events_json,encoder_sha256 FROM neural_pointer_runs').fetchall()
        assert len(rows) == 3 and all(len(json.loads(r[0])) == 641 for r in rows)
        assert conn.execute('SELECT keystroke_model FROM users').fetchone()[0] is None
    setup(client, monkeypatch, 'second-user')
    result = client.get('/api/pointer/status').json()
    assert not result['enrolled'] and result['count'] == 0


def test_nonce_cannot_be_used_by_another_account(client, monkeypatch):
    _, clock = setup(client, monkeypatch)
    token = client.post('/api/pointer/challenge', json={'kind': 'enroll'}).json()['id']
    _, clock = setup(client, monkeypatch, 'second-user')
    clock[0] += 61
    assert client.post('/api/pointer/capture', json={'kind': 'enroll', 'challenge_id': token, 'events': events()}).status_code == 409


@pytest.mark.parametrize('fault', ['expired', 'short', 'untrusted', 'touch', 'impossible_clock', 'stationary'])
def test_invalid_recordings_do_not_enroll(client, monkeypatch, fault):
    _, clock = setup(client, monkeypatch)
    token = client.post('/api/pointer/challenge', json={'kind': 'enroll'}).json()['id']
    e = events()
    clock[0] += 61
    if fault == 'expired': clock[0] += 200
    if fault == 'short': e = e[:100]
    if fault == 'untrusted': e[0]['trusted'] = False
    if fault == 'touch': e[0]['pointer_type'] = 'touch'
    if fault == 'impossible_clock': clock[0] -= 60
    if fault == 'stationary':
        for event in e:
            event['x'], event['y'] = 0, 0
    r = client.post('/api/pointer/capture', json={'kind': 'enroll', 'challenge_id': token, 'events': e})
    assert r.status_code in (409, 422)
    assert client.get('/api/pointer/status').json()['count'] == 0


def test_replay_is_rejected_even_with_new_nonce(client, monkeypatch):
    _, clock = setup(client, monkeypatch)
    for index in range(2):
        token = client.post('/api/pointer/challenge', json={'kind': 'enroll'}).json()['id']
        clock[0] += 61
        r = client.post('/api/pointer/capture', json={'kind': 'enroll', 'challenge_id': token, 'events': events()})
        assert r.status_code == (200 if index == 0 else 422)


@pytest.mark.parametrize('matches', [True, False])
def test_trained_profile_controls_stepup_and_session(client, monkeypatch, matches):
    fake, clock = setup(client, monkeypatch)
    enroll(client, clock)
    pending()
    client.cookies.clear()
    body = {'kind': 'verify', 'username': 'pointer-user', 'password': 'test-only'}
    assert client.post('/api/pointer/challenge', json={**body, 'password': 'wrong'}).status_code == 401
    token = client.post('/api/pointer/challenge', json=body).json()['id']
    if not matches:
        fake.encode = lambda events: np.tile(np.r_[0., 1., np.zeros(126)], (5, 1))
    clock[0] += 11
    payload = {**body, 'challenge_id': token, 'events': events(20, 10)}
    r = client.post('/api/pointer/capture', json=payload)
    assert r.status_code == 200, r.text
    assert r.json()['decision'] == ('allow' if matches else 'block')
    assert r.json()['signals'][-1]['name'] == 'pointer_neural'
    assert client.get('/api/session').status_code == (200 if matches else 401)
    assert client.post('/api/pointer/capture', json=payload).status_code == 409


def test_pointer_check_cannot_be_used_without_pending_login(client, monkeypatch):
    _, clock = setup(client, monkeypatch)
    enroll(client, clock)
    assert client.post('/api/pointer/challenge', json={'kind': 'verify', 'username': 'pointer-user', 'password': 'test-only'}).status_code == 409


def test_legacy_keypad_cannot_bypass_neural_requirement(client, monkeypatch):
    _, clock = setup(client, monkeypatch)
    enroll(client, clock)
    import db
    with db.connect() as conn:
        conn.execute("UPDATE users SET keypad_model='{}'")
    pending()
    # Obtain a valid-shaped run using the existing test helper; rejection must
    # occur at the pending-route guard, before keypad feature scoring.
    from test_keypad import run as make_run
    run = make_run().model_dump()
    r = client.post('/api/login/keypad', json={'username': 'pointer-user', 'password': 'test-only', 'runs': [run, run]})
    assert r.status_code == 409
    assert 'trained pointer' in r.json()['detail']
