"""Rejected submissions are kept: correction habits are data."""

import sqlite3

from helpers import PASSWORD, make_sample

TYPO = ["KeyT", "KeyO", "Backspace", "KeyI", "KeyE", "Digit5"]


def _rows(client):
    import db
    with sqlite3.connect(db.DB) as c:
        return c.execute("SELECT kind, status, corrections, keystroke_vector IS NOT NULL FROM samples ORDER BY id").fetchall()


def test_rejected_enrollment_reps_are_stored_but_not_counted(client):
    client.post("/api/register", json={"username": "a", "password": PASSWORD})
    post = lambda pw, s: client.post("/api/enroll", json={"username": "a", "password": pw, "sample": s}).json()
    assert not post(PASSWORD, make_sample(codes=TYPO, seed=1))["accepted"]
    assert not post("nope", make_sample(seed=2))["accepted"]
    r = post(PASSWORD, make_sample(seed=3))
    assert r["accepted"] and r["warmup"] and r["count"] == 0
    assert _rows(client) == [("enroll", "retype", 1, 0), ("enroll", "wrong_password", 0, 0), ("enroll", "ok", 0, 1)]


def test_wrong_password_and_retype_logins_are_stored_and_linked(client):
    client.post("/api/register", json={"username": "a", "password": PASSWORD})
    for i in range(11):
        client.post("/api/enroll", json={"username": "a", "password": PASSWORD, "sample": make_sample(seed=i)})
    login = lambda pw, s: client.post("/api/login", json={"username": "a", "password": pw, "sample": s}).json()
    a = login("nope", make_sample(seed=50))
    b = login(PASSWORD, make_sample(codes=TYPO, seed=51))
    assert a["decision"] == "wrong_password" and b["decision"] == "retype"
    attempts = {r["id"]: r for r in client.get("/api/attempts", params={"user": "a"}).json()}
    assert attempts[a["attempt_id"]]["sample_id"] is not None
    assert attempts[b["attempt_id"]]["sample_id"] is not None
    assert _rows(client)[-2:] == [("login", "wrong_password", 0, 0), ("login", "retype", 1, 0)]

    habits = client.get("/api/users/a").json()["habits"]
    assert habits["submissions"] == 13
    assert habits["corrected"] == 1 and habits["wrong_password"] == 1
    assert abs(habits["correction_rate"] - 1 / 13) < 1e-9


def test_migration_adds_columns_to_an_old_database(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as c:
        c.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, user_id INTEGER, kind TEXT, created_at TEXT,"
                  " sample_json TEXT, keystroke_vector TEXT, pointer_vector TEXT)")
    monkeypatch.setenv("BIOPRINT_DB", str(old))
    import importlib

    import db
    importlib.reload(db)
    db.init_db()
    with sqlite3.connect(old) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(samples)")}
    assert {"status", "corrections"} <= cols


def test_pasted_password_asks_to_type_it(client):
    client.post("/api/register", json={"username": "a", "password": PASSWORD})
    for i in range(11):
        client.post("/api/enroll", json={"username": "a", "password": PASSWORD, "sample": make_sample(seed=i)})
    from test_bot import human_env
    pasted = make_sample(codes=["ControlLeft", "KeyV"], seed=7, env=human_env())
    pasted["meta"]["had_paste"] = True
    r = client.post("/api/login", json={"username": "a", "password": PASSWORD, "sample": pasted}).json()
    assert r["decision"] == "retype" and "pasted" in r["reasons"][0]
