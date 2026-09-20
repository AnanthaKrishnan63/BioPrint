"""Each repetition's password is checked before the browser retains timing."""
import json
import pytest
from helpers import PASSWORD, make_sample
from test_bot import human_env


def pending(client):
    creds = {'username': 'repetition-check', 'password': PASSWORD}
    assert client.post('/api/register', json=creds).status_code == 200
    env = human_env(fonts=['Arial', 'Synthetic'], webgl_renderer='NVIDIA GeForce RTX 3060')
    for i in range(11):
        r = client.post('/api/enroll', json={**creds, 'sample': make_sample(seed=i, env=env)}).json()
        assert r['accepted'], r
    # Create a pending attempt with known evidence: this test covers repetition
    # checking, independently of which branch routes to that phase.
    import db
    with db.connect() as conn:
        user = conn.execute('SELECT * FROM users WHERE username=?', (creds['username'],)).fetchone()
        model = json.loads(user['keystroke_model'])
        conn.execute('INSERT INTO attempts (user_id, username, created_at, decision, reasons, signals, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (user['id'], creds['username'], db.now(), 'step_up', '[]',
                      json.dumps([dict(name='keystroke', score=.3, threshold=model['threshold'], available=True)]), 0.))
        attempt_id = conn.execute('SELECT max(id) FROM attempts').fetchone()[0]
    return creds, env, attempt_id


def test_wrong_password_is_immediate_and_does_not_change_pending_state(client):
    creds, env, original = pending(client)
    for index, password in enumerate([PASSWORD, 'incorrect', PASSWORD]):
        r = client.post('/api/login/stepup/check', json={**creds, 'password': password,
                                                       'sample': make_sample(seed=100+index, env=env)})
        assert r.status_code == 200
        assert r.json()['accepted'] is (password == PASSWORD)
        if password != PASSWORD:
            assert 'Incorrect password' in r.json()['reason']
        assert 'set-cookie' not in r.headers
    import db
    with db.connect() as conn:
        assert conn.execute('SELECT max(id) FROM attempts').fetchone()[0] == original
        assert conn.execute("SELECT count(*) FROM samples WHERE kind='login'").fetchone()[0] == 0
    assert client.get('/api/session').status_code == 401


@pytest.mark.parametrize('completed', [False, True])
def test_missing_or_completed_stepup_is_rejected(client, completed):
    creds, env, original = pending(client)
    import db
    with db.connect() as conn:
        if completed:
            conn.execute("UPDATE attempts SET decision='allow' WHERE id=?", (original,))
        else:
            conn.execute("UPDATE attempts SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (original,))
    r = client.post('/api/login/stepup/check', json={**creds, 'sample': make_sample(seed=99, env=env)})
    assert r.status_code == 409


def test_correct_password_with_unusual_behavior_is_kept_for_final_scoring(client):
    creds, env, _ = pending(client)
    sample = make_sample(seed=101, env=env, hold=220, gap=400, trusted=False)
    # Credential checking must not cherry-pick favorable biometric scores.
    r = client.post('/api/login/stepup/check', json={**creds, 'sample': sample})
    assert r.json() == {'accepted': True}
    r = client.post('/api/login/stepup', json={**creds, 'samples': [sample]*3})
    assert r.json()['decision'] == 'block'


def test_unusable_sample_requests_only_that_entry_again(client):
    creds, env, _ = pending(client)
    sample = make_sample(seed=102, env=env)
    sample['keystrokes'] = []
    r = client.post('/api/login/stepup/check', json={**creds, 'sample': sample})
    assert not r.json()['accepted']
    assert 'Repeat only this entry' in r.json()['reason']
