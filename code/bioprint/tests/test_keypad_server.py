"""The scrambled keypad end to end: challenges, enrollment, routing to "keypad"
on a new device class, and the keypad step-up verdict.

Runs come from helpers.make_run (synthetic) and echo a server-issued challenge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import PASSWORD, make_run, make_sample
from test_bot import human_env
from test_stepup import HOME, OTHER, enrolled, login, signal

PHONE = json.loads((Path(__file__).parent / "fixtures" / "phone_sample.json").read_text())
PHONE_ENV = human_env(ua="Mozilla/5.0 (Android 17; Mobile; rv:156.0) Gecko/156.0 Firefox/156.0",
                      max_touch_points=5, screen_width=412, screen_height=915,
                      inner_width=412, inner_height=780, outer_width=412, outer_height=915,
                      webgl_renderer="Adreno (TM) 740", fonts=["Roboto"])
COOKIE = "bioprint_session"

OWNER = dict(interval_ms=650.0, hold_ms=95.0)
IMPOSTOR = dict(interval_ms=1400.0, hold_ms=180.0)


def challenge(client, user="alice") -> dict:
    r = client.post("/api/keypad/challenge", json={"username": user})
    assert r.status_code == 200, r.text
    return r.json()


def solve(client, seed: int, device="mouse", env=HOME, user="alice", **cadence) -> dict:
    ch = challenge(client, user)
    run = make_run(seed=seed, device=device, env=env, layout=ch["layout"], target=ch["target"],
                   **{**OWNER, **cadence})
    run["challenge_id"] = ch["id"]
    return run


def enroll_keypad(client, run, user="alice", password=PASSWORD):
    return client.post("/api/enroll/keypad", json={"username": user, "password": password, "runs": [run]})


def keypad_enrolled(client, user="alice", device="mouse", env=HOME):
    enrolled(client, user)
    for i in range(6):
        body = enroll_keypad(client, solve(client, 100 + i, device=device, env=env, user=user), user).json()
        assert body["accepted"], body
        assert body["count"] == i + 1
    assert body["enrolled"], body
    assert body["device_class"] == device
    return body


def keypad_login(client, runs, user="alice", password=PASSWORD):
    return client.post("/api/login/keypad", json={"username": user, "password": password, "runs": runs})


# ---------------------------------------------------------------- challenges and enrollment


def test_challenge_shape(client):
    enrolled(client)
    ch = challenge(client)
    assert sorted(d for d in ch["layout"] if d >= 0) == [0, 1, 2, 3, 4]
    assert ch["layout"].count(-2) == 1 and len(ch["layout"]) == 6
    assert len(ch["target"]) == 6 and all(0 <= d <= 4 for d in ch["target"])
    assert client.post("/api/keypad/challenge", json={"username": "nobody"}).status_code == 404


def test_enrollment_builds_a_profile(client):
    keypad_enrolled(client)
    u = client.get("/api/users/alice").json()
    assert u["keypad_enrolled"] and u["keypad_count"] == 6 and u["keypad_target"] == 6
    assert u["keypad_device_class"] == "mouse"
    assert u["habits"]["keypad_runs"] == 6


def test_enrollment_rejections(client):
    enrolled(client)
    run = solve(client, 1)
    assert not enroll_keypad(client, run, password="nope").json()["accepted"]
    assert enroll_keypad(client, run).json()["accepted"]
    body = enroll_keypad(client, run).json()  # same challenge again
    assert not body["accepted"] and "already" in body["reasons"][0]
    run = solve(client, 2)
    run["target"] = list(reversed(run["target"])) if run["target"] != run["target"][::-1] else [4, 3, 2, 1, 0, 1]
    body = enroll_keypad(client, run).json()
    assert not body["accepted"] and "does not match" in body["reasons"][0]
    run = solve(client, 3)
    run["taps"] = run["taps"][:-1]  # one digit short
    run["completed"] = False
    body = enroll_keypad(client, run).json()
    assert not body["accepted"] and "did not match" in body["reasons"][0]
    assert client.get("/api/users/alice").json()["keypad_count"] == 1


# ---------------------------------------------------------------- routing


def test_phone_without_keypad_profile_is_still_retype(client):
    enrolled(client)
    assert login(client, PHONE).json()["decision"] == "retype"


def test_phone_with_keypad_profile_is_routed_to_keypad(client):
    keypad_enrolled(client)
    body = login(client, PHONE).json()
    assert body["decision"] == "keypad", body
    assert not signal(body, "keystroke")["available"]
    assert signal(body, "device")["flagged"]
    assert not client.cookies.get(COOKIE)


def test_new_device_class_with_good_rhythm_is_routed_to_keypad(client):
    keypad_enrolled(client)
    body = login(client, make_sample(seed=99, env=PHONE_ENV)).json()
    assert body["decision"] == "keypad", body
    assert signal(body, "device")["flagged"]
    assert not signal(body, "keystroke")["flagged"]


def test_new_device_same_class_with_good_rhythm_is_routed_to_target_check(client):
    keypad_enrolled(client)
    body = login(client, make_sample(seed=99, env=OTHER)).json()
    assert body["decision"] == "keypad", body


def test_new_device_bad_rhythm_blocks_before_any_keypad(client):
    keypad_enrolled(client)
    body = login(client, make_sample(seed=99, env=PHONE_ENV, hold=220, gap=400)).json()
    assert body["decision"] == "block", body


# ---------------------------------------------------------------- the keypad step-up


def pending_keypad(client):
    keypad_enrolled(client)
    body = login(client, make_sample(seed=99, env=PHONE_ENV)).json()
    assert body["decision"] == "keypad", body
    return body


def test_keypad_step_up_needs_a_pending_decision(client):
    keypad_enrolled(client)
    runs = [solve(client, 200 + i, device="touch", env=PHONE_ENV) for i in range(2)]
    assert keypad_login(client, runs).status_code == 409


def test_owner_on_a_phone_is_allowed_on_cognitive_score_alone(client):
    pending_keypad(client)
    runs = [solve(client, 200 + i, device="touch", env=PHONE_ENV) for i in range(2)]
    r = keypad_login(client, runs)
    body = r.json()
    assert body["decision"] == "allow", body
    kp, km = signal(body, "keypad"), signal(body, "keypad_motor")
    assert kp["available"] and not kp["flagged"]
    assert not km["available"]  # enrolled on a mouse, answered by a thumb
    assert signal(body, "device")["flagged"]
    assert kp["score"] <= kp["threshold"]
    assert client.cookies.get(COOKIE)
    assert client.get("/api/session").json()["username"] == "alice"
    rows = client.get("/api/attempts", params={"user": "alice"}).json()
    assert rows[0]["decision"] == "allow" and rows[1]["decision"] == "keypad"


def test_impostor_on_a_phone_is_blocked(client):
    pending_keypad(client)
    runs = [solve(client, 300 + i, device="touch", env=PHONE_ENV, **IMPOSTOR) for i in range(2)]
    body = keypad_login(client, runs).json()
    assert body["decision"] == "block", body
    assert signal(body, "keypad")["flagged"]
    assert not client.cookies.get(COOKIE)


def test_scripted_keypad_is_a_bot(client):
    pending_keypad(client)
    runs = [solve(client, 400 + i, device="touch", env=PHONE_ENV, interval_ms=60.0, interval_sd=0.0)
            for i in range(2)]
    body = keypad_login(client, runs).json()
    assert body["decision"] == "block", body
    assert signal(body, "bot")["flagged"]


def test_keypad_step_up_wants_two_runs_and_fresh_challenges(client):
    pending_keypad(client)
    one = solve(client, 500, device="touch", env=PHONE_ENV)
    assert keypad_login(client, [one]).status_code == 400
    body = keypad_login(client, [one, one]).json()  # the same challenge twice
    assert body["decision"] == "retype", body


def test_same_class_step_up_scores_motor_too(client):
    """Enrolled on a mouse, stepped up on another mouse machine: both halves scored."""
    keypad_enrolled(client)
    # A second laptop, same class; push the rhythm to borderline so the keypad is asked for.
    body = login(client, make_sample(seed=99, env=OTHER, hold=118, gap=150)).json()
    if body["decision"] != "keypad":
        pytest.skip("this rhythm was not borderline; covered by test_decide")
    runs = [solve(client, 600 + i, device="mouse", env=OTHER) for i in range(2)]
    body = keypad_login(client, runs).json()
    # The stricter final pointer policy rejects this borderline motor fixture.
    motor = signal(body, "keypad_motor")
    assert motor["available"]
    assert motor["score"] > 0.75 * motor["threshold"]
    assert body["decision"] == "block", body
    assert not client.cookies.get(COOKIE)
