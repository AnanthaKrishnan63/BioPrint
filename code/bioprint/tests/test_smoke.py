"""End-to-end through the HTTP API. Proves the contracts hold together; the real
accuracy tests live with each engine module."""

from helpers import CODES, PASSWORD, make_sample


def _enrolled(client, user="alice"):
    assert client.post("/api/register", json={"username": user, "password": PASSWORD}).status_code == 200
    for i in range(10):
        r = client.post("/api/enroll", json={"username": user, "password": PASSWORD,
                                             "sample": make_sample(seed=i)}).json()
        assert r["accepted"], r
    assert r["enrolled"]


def login(client, sample, password=PASSWORD, user="alice"):
    return client.post("/api/login", json={"username": user, "password": password, "sample": sample}).json()


def test_genuine_user_is_allowed(client):
    _enrolled(client)
    r = login(client, make_sample(seed=99))
    assert r["decision"] == "allow", r
    assert r["latency_ms"] < 500


def test_different_rhythm_is_blocked(client):
    _enrolled(client)
    r = login(client, make_sample(hold=220, gap=400, seed=5))
    assert r["decision"] == "block", r


def test_script_events_are_blocked_as_bot(client):
    _enrolled(client)
    r = login(client, make_sample(seed=99, trusted=False))
    assert r["decision"] == "block"
    assert any(s["name"] == "bot" and s["flagged"] for s in r["signals"])


def test_wrong_password_and_retype(client):
    _enrolled(client)
    assert login(client, make_sample(seed=1), password="nope")["decision"] == "wrong_password"
    typo = make_sample(codes=["KeyT", "KeyO", "Backspace", "KeyI", "KeyE", "Digit5"], seed=1)
    assert login(client, typo)["decision"] == "retype"


def test_not_enrolled_and_unknown(client):
    client.post("/api/register", json={"username": "bob", "password": PASSWORD})
    assert login(client, make_sample(), user="bob")["decision"] == "not_enrolled"
    assert login(client, make_sample(), user="nobody")["decision"] == "unknown_user"


def test_attempts_are_logged(client):
    _enrolled(client)
    login(client, make_sample(seed=99))
    rows = client.get("/api/attempts", params={"user": "alice"}).json()
    assert rows and rows[0]["decision"] == "allow"


def test_attempts_can_be_labelled(client):
    _enrolled(client)
    aid = login(client, make_sample(seed=99))["attempt_id"]
    assert client.post(f"/api/attempts/{aid}/label", json={"label": "genuine"}).status_code == 200
    assert client.get("/api/attempts").json()[0]["label"] == "genuine"
    assert client.post(f"/api/attempts/{aid}/label", json={"label": "maybe"}).status_code == 400
