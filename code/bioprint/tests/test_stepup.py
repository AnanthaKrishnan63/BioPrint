"""Step-up: password right, rhythm right, device new -> ask for three more typings
and judge the median of the four keystroke scores. Same evidence, more of it.

    device same  + keystroke pass -> allow
    device same  + keystroke fail -> block
    device NEW   + keystroke pass -> step_up  (then median of 4 -> allow / block)
    device NEW   + keystroke fail -> block
"""

from __future__ import annotations

import pytest
from helpers import PASSWORD, make_sample
from test_bot import human_env

COOKIE = "bioprint_session"
FONTS = ["Arial", "DejaVu Sans", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans", "Ubuntu"]

HOME = human_env(fonts=list(FONTS))
# fonts 13.9 + time zone 3.04 + GPU 3.4 = 20.3 bits: far over the 6-bit limit.
OTHER = human_env(fonts=["Arial", "Calibri", "Segoe UI", "Tahoma"], timezone="Europe/London",
                  webgl_renderer="ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))")

IMPOSTOR = dict(hold=220, gap=400)


def owner(seed: int, env: dict = HOME, **over) -> dict:
    return make_sample(seed=seed, env=env, **over)


def enrolled(client, user="alice"):
    assert client.post("/api/register", json={"username": user, "password": PASSWORD}).status_code == 200
    for i in range(11):  # 1 practice run + 10 counted
        r = client.post("/api/enroll", json={"username": user, "password": PASSWORD, "sample": owner(i)}).json()
        assert r["accepted"], r
    assert r["enrolled"]


def login(client, sample, password=PASSWORD, user="alice"):
    return client.post("/api/login", json={"username": user, "password": password, "sample": sample})


def step_up(client, samples, password=PASSWORD, user="alice"):
    return client.post("/api/login/stepup", json={"username": user, "password": password, "samples": samples})


def signal(body: dict, name: str) -> dict:
    return next(s for s in body["signals"] if s["name"] == name)


def pending_step_up(client) -> dict:
    """Enrolled at HOME, then the owner's rhythm from OTHER: must be step_up, not a login."""
    enrolled(client)
    r = login(client, owner(99, env=OTHER))
    body = r.json()
    assert body["decision"] == "step_up", body
    assert "set-cookie" not in r.headers
    assert not client.cookies.get(COOKIE)
    assert client.get("/api/session").status_code == 401
    return body


# ---------------------------------------------------------------- the table


def test_same_device_owner_rhythm_is_allowed(client):
    enrolled(client)
    body = login(client, owner(99)).json()
    assert body["decision"] == "allow", body
    assert signal(body, "device")["available"] and not signal(body, "device")["flagged"]


def test_new_device_owner_rhythm_asks_for_step_up(client):
    body = pending_step_up(client)
    dv, ks = signal(body, "device"), signal(body, "keystroke")
    assert dv["available"] and dv["flagged"] and dv["score"] > 6
    assert not ks["flagged"]
    assert any("new device" in r for r in body["reasons"])
    assert any("installed fonts changed" in r for r in body["reasons"])  # device reasons are carried
    rows = client.get("/api/attempts", params={"user": "alice"}).json()
    assert rows[0]["decision"] == "step_up" and rows[0]["id"] == body["attempt_id"]


def test_new_device_and_impostor_rhythm_blocks_directly(client):
    enrolled(client)
    r = login(client, owner(5, env=OTHER, **IMPOSTOR))
    body = r.json()
    assert body["decision"] == "block", body
    assert signal(body, "keystroke")["flagged"] and signal(body, "device")["flagged"]
    assert any("different device" in x for x in body["reasons"])
    assert not client.cookies.get(COOKIE)


def test_device_unavailable_counts_as_same_device(client):
    """No probe on this attempt: nothing to ask more about, the rhythm decides."""
    enrolled(client)
    assert login(client, owner(99, env={})).json()["decision"] == "allow"
    assert login(client, owner(5, env={}, **IMPOSTOR)).json()["decision"] == "block"


# ---------------------------------------------------------------- answering the step-up


def test_step_up_with_owner_samples_allows_and_signs_in(client):
    first = pending_step_up(client)
    r = step_up(client, [owner(101, env=OTHER), owner(102, env=OTHER), owner(103, env=OTHER)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "allow", body
    assert set(body) == {"decision", "reasons", "signals", "latency_ms", "attempt_id"}
    assert body["reasons"][0].startswith("new device: median of 4 samples")
    assert COOKIE in r.headers["set-cookie"] and client.cookies.get(COOKIE)
    assert client.get("/api/session").json()["username"] == "alice"

    ks = signal(body, "keystroke")
    assert not ks["flagged"] and ks["threshold"] == signal(first, "keystroke")["threshold"]
    assert "median of 4 samples" in ks["reasons"][0]
    assert {s["name"] for s in body["signals"]} == {"keystroke", "device", "bot", "pointer"}
    assert signal(body, "device")["flagged"]  # still a new device, now advisory

    rows = client.get("/api/attempts", params={"user": "alice"}).json()
    assert [x["decision"] for x in rows[:2]] == ["allow", "step_up"]
    assert rows[0]["latency_ms"] > 0
    stored = client.get("/api/users/alice").json()["habits"]
    assert stored["submissions"] == 11 + 1 + 3  # every sample is kept


def test_step_up_with_impostor_samples_blocks_and_clears_session(client):
    enrolled(client)
    assert login(client, owner(98)).json()["decision"] == "allow"  # a session to lose
    assert client.get("/api/session").status_code == 200
    r = login(client, owner(99, env=OTHER))
    assert r.json()["decision"] == "step_up"
    assert client.get("/api/session").status_code == 200  # step_up leaves the cookie alone

    r = step_up(client, [owner(s, env=OTHER, **IMPOSTOR) for s in (11, 12, 13)])
    body = r.json()
    assert body["decision"] == "block", body
    assert signal(body, "keystroke")["flagged"]
    assert body["reasons"][0].startswith("new device: median of 4 samples")
    assert not client.cookies.get(COOKIE)
    assert client.get("/api/session").status_code == 401


def test_median_is_of_all_four(client):
    """The first (owner) sample is one of four: one impostor sample among three owner
    ones is outvoted; the reported score is the median, not the mean."""
    pending_step_up(client)
    body = step_up(client, [owner(101, env=OTHER), owner(102, env=OTHER, **IMPOSTOR), owner(103, env=OTHER)]).json()
    assert body["decision"] == "allow", body
    ks = signal(body, "keystroke")
    assert ks["score"] < ks["threshold"]
    assert "(" in ks["reasons"][0] and ks["reasons"][0].count(",") == 3  # four scores listed


def test_step_up_without_pending_attempt_is_409(client):
    enrolled(client)
    r = step_up(client, [owner(s) for s in (101, 102, 103)])
    assert r.status_code == 409, r.text
    assert client.get("/api/session").status_code == 401


def test_step_up_answered_once_cannot_be_answered_again(client):
    pending_step_up(client)
    assert step_up(client, [owner(s, env=OTHER) for s in (101, 102, 103)]).json()["decision"] == "allow"
    assert step_up(client, [owner(s, env=OTHER) for s in (104, 105, 106)]).status_code == 409


def test_step_up_expires(client, monkeypatch):
    import server

    pending_step_up(client)
    monkeypatch.setattr(server, "STEP_UP_WINDOW_S", -60)
    assert step_up(client, [owner(s, env=OTHER) for s in (101, 102, 103)]).status_code == 409


def test_bot_sample_in_the_batch_blocks(client):
    pending_step_up(client)
    batch = [owner(101, env=OTHER), owner(102, env=OTHER, trusted=False), owner(103, env=OTHER)]
    body = step_up(client, batch).json()
    assert body["decision"] == "block", body
    assert signal(body, "bot")["flagged"]
    assert not client.cookies.get(COOKIE)


def test_corrected_sample_in_the_batch_means_retype_and_keeps_the_step_up_pending(client):
    pending_step_up(client)
    typo = owner(102, env=OTHER, codes=["KeyT", "KeyO", "Backspace", "KeyI", "KeyE", "Digit5"])
    body = step_up(client, [owner(101, env=OTHER), typo, owner(103, env=OTHER)]).json()
    assert body["decision"] == "retype", body
    assert "1 of the 3" in body["reasons"][0] and "all 3" in body["reasons"][0]
    assert signal(body, "device")["available"]
    assert not client.cookies.get(COOKIE)
    # still pending: a clean batch now succeeds
    assert step_up(client, [owner(s, env=OTHER) for s in (104, 105, 106)]).json()["decision"] == "allow"


def test_step_up_wrong_password_and_unknown_user(client):
    pending_step_up(client)
    r = step_up(client, [owner(s, env=OTHER) for s in (101, 102, 103)], password="nope")
    assert r.json()["decision"] == "wrong_password" and "set-cookie" not in r.headers
    r = step_up(client, [owner(s, env=OTHER) for s in (101, 102, 103)], user="nobody")
    assert r.json()["decision"] == "unknown_user"
    assert client.get("/api/session").status_code == 401


def test_step_up_needs_exactly_three_samples(client):
    pending_step_up(client)
    assert step_up(client, [owner(101, env=OTHER)]).status_code == 422
    assert step_up(client, [owner(s, env=OTHER) for s in range(101, 105)]).status_code == 422


# ---------------------------------------------------------------- device on retype


def test_retype_attempt_still_carries_the_device_signal(client):
    """Mobile keyboards send an empty event.code, so a phone login is a retype today;
    the device axis must still be measured so it can be shown from a phone."""
    enrolled(client)
    typo = owner(1, env=OTHER, codes=["KeyT", "KeyO", "Backspace", "KeyI", "KeyE", "Digit5"])
    body = login(client, typo).json()
    assert body["decision"] == "retype", body
    dv = signal(body, "device")
    assert dv["available"] and dv["flagged"] and dv["score"] == pytest.approx(13.9 + 3.04 + 3.4)
    assert not any(s["name"] == "keystroke" for s in body["signals"])  # rhythm itself is not scored
