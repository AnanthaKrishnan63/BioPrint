"""The session cookie: an "allow" signs the browser in, a "block" signs it out.
Goes through the HTTP API like test_smoke.py; the TestClient keeps a cookie jar."""

from helpers import PASSWORD, make_sample

COOKIE = "bioprint_session"


def _enrolled(client, user="alice"):
    assert client.post("/api/register", json={"username": user, "password": PASSWORD}).status_code == 200
    for i in range(11):  # 1 practice run + 10 counted
        r = client.post("/api/enroll", json={"username": user, "password": PASSWORD,
                                             "sample": make_sample(seed=i)}).json()
        assert r["accepted"], r
    assert r["enrolled"]


def login(client, sample, password=PASSWORD, user="alice"):
    return client.post("/api/login", json={"username": user, "password": password, "sample": sample})


def test_no_session_before_login(client):
    assert client.get("/api/session").status_code == 401


def test_allow_sets_cookie_and_session_works(client):
    _enrolled(client)
    r = login(client, make_sample(seed=99))
    assert r.json()["decision"] == "allow", r.json()
    set_cookie = r.headers["set-cookie"]
    assert COOKIE in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()
    assert client.cookies.get(COOKIE)

    s = client.get("/api/session")
    assert s.status_code == 200
    assert s.json()["username"] == "alice"
    assert s.json()["since"]


def test_tampered_cookie_is_rejected(client):
    _enrolled(client)
    assert login(client, make_sample(seed=99)).json()["decision"] == "allow"
    token = client.cookies.get(COOKIE)
    body, sig = token.rsplit(".", 1)
    forged = body + "." + ("0" if sig[0] != "0" else "1") + sig[1:]
    for bad in (forged, "garbage", "a.b", ""):
        client.cookies.set(COOKIE, bad)
        assert client.get("/api/session").status_code == 401, bad
    client.cookies.set(COOKIE, token)
    assert client.get("/api/session").status_code == 200


def test_block_clears_session(client):
    _enrolled(client)
    assert login(client, make_sample(seed=99)).json()["decision"] == "allow"
    assert client.get("/api/session").status_code == 200

    r = login(client, make_sample(hold=220, gap=400, seed=5))
    assert r.json()["decision"] == "block", r.json()
    assert not client.cookies.get(COOKIE)
    assert client.get("/api/session").status_code == 401


def test_logout_clears_session(client):
    _enrolled(client)
    assert login(client, make_sample(seed=99)).json()["decision"] == "allow"
    assert client.post("/api/logout").status_code == 200
    assert not client.cookies.get(COOKIE)
    assert client.get("/api/session").status_code == 401


def test_wrong_password_sets_nothing(client):
    _enrolled(client)
    r = login(client, make_sample(seed=1), password="nope")
    assert r.json()["decision"] == "wrong_password"
    assert "set-cookie" not in r.headers
    assert not client.cookies.get(COOKIE)
    assert client.get("/api/session").status_code == 401


def test_other_non_allow_decisions_set_nothing(client):
    client.post("/api/register", json={"username": "bob", "password": PASSWORD})
    for user in ("bob", "nobody"):
        r = login(client, make_sample(), user=user)
        assert r.json()["decision"] in ("not_enrolled", "unknown_user")
        assert "set-cookie" not in r.headers
    assert client.get("/api/session").status_code == 401


def test_login_response_shape_unchanged(client):
    """LoginOut is a contract: the cookie must not leak into the body."""
    _enrolled(client)
    body = login(client, make_sample(seed=99)).json()
    assert set(body) == {"decision", "reasons", "signals", "latency_ms", "attempt_id"}
