"""A phone login is a human on another device class, not a bot. Fixture is a real
Android Firefox attempt (attempt #34 on the demo box): empty key codes, 0 ms holds."""

import json
from pathlib import Path

from helpers import PASSWORD, make_sample
from test_bot import human_env

PHONE = json.loads((Path(__file__).parent / "fixtures" / "phone_sample.json").read_text())


def _enrolled(client, user="a"):
    client.post("/api/register", json={"username": user, "password": PASSWORD})
    for i in range(11):
        client.post("/api/enroll", json={"username": user, "password": PASSWORD,
                                         "sample": make_sample(seed=i, env=human_env())})


def test_touch_keyboard_is_retype_not_bot_and_carries_the_device_signal(client):
    _enrolled(client)
    r = client.post("/api/login", json={"username": "a", "password": PASSWORD, "sample": PHONE}).json()
    assert r["decision"] == "retype", r
    assert "touch keyboard" in r["reasons"][0]
    sig = {s["name"]: s for s in r["signals"]}
    assert not sig["bot"]["flagged"], sig["bot"]["reasons"]
    assert sig["device"]["available"] and sig["device"]["flagged"], sig["device"]
    assert sig["device"]["score"] > 6


def test_every_attempt_carries_the_device_signal_even_when_a_bot_is_caught(client):
    _enrolled(client)
    bot = make_sample(seed=99, trusted=False, env=human_env())
    r = client.post("/api/login", json={"username": "a", "password": PASSWORD, "sample": bot}).json()
    assert r["decision"] == "block"
    assert {s["name"] for s in r["signals"]} >= {"device", "bot"}
