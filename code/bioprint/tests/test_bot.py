"""engine/bot.py: every rule fires on its own attack and stays quiet on humans.

The "human" baselines here are deliberately unflattering to the detector: fast
typists, rollover (a key pressed before the previous is released), a Linux box with
llvmpipe software rendering, a submit by Enter with no pointer at all.
"""

from __future__ import annotations

import random

import pytest
from helpers import CODES, make_sample

from contracts import FeatureVector, Sample
from engine import bot, features

LONG_CODES = ["Period", "KeyT", "KeyI", "KeyE", "Digit5", "KeyR", "KeyO", "KeyA", "KeyN", "KeyL"]  # .tie5Roanl


def human_env(**over) -> dict:
    env = {
        "probe_version": 1, "webdriver": False,
        "ua": "Mozilla/5.0 (X11; Linux x86_64; rv:155.0) Gecko/20100101 Firefox/155.0",
        "ua_brands": None, "ua_mobile": None, "platform": "Linux x86_64",
        "plugins": 5, "mime_types": 2, "languages": ["en-GB", "en"],
        "outer_width": 1920, "outer_height": 1050, "inner_width": 1920, "inner_height": 950,
        "screen_width": 1920, "screen_height": 1080, "hardware_concurrency": 8, "device_memory": 8,
        "max_touch_points": 0, "has_window_chrome": False, "notification_permission": "default",
        "permissions_notifications": "prompt", "timezone": "Asia/Kolkata", "automation_globals": [],
        "webgl_vendor": "Mesa", "webgl_renderer": "llvmpipe (LLVM 17.0.6, 256 bits)",  # real Linux VM
    }
    env.update(over)
    return env


_DEFAULT_ENV = object()


def human(seed: int = 1, codes=LONG_CODES, hold=95.0, gap=170.0, jitter=35.0, env=_DEFAULT_ENV, **over) -> Sample:
    d = make_sample(codes=codes, hold=hold, gap=gap, jitter=jitter, seed=seed,
                    env=human_env() if env is _DEFAULT_ENV else env)
    d.update(over)
    return Sample.model_validate(d)


def check(sample: Sample, vector=None, priors=(), password_length=10) -> "bot.SignalResult":
    return bot.check(sample, vector, list(priors), password_length)


def vec(sample: Sample) -> FeatureVector:
    return features.keystroke_vector(features.password_keystrokes(sample))


def fired(result) -> set[str]:
    return {c.feature.removeprefix("bot.") for c in result.contributions}


# ---------------------------------------------------------------- humans pass


@pytest.mark.parametrize("seed", range(12))
def test_ordinary_human_is_not_flagged(seed):
    r = check(human(seed))
    assert not r.flagged, (r.score, r.reasons)
    assert fired(r) <= {"software_gl"}


@pytest.mark.parametrize("seed", range(6))
def test_fast_typist_with_rollover_is_not_flagged(seed):
    """80 ms between keys and holds long enough to overlap the next press (UD < 0):
    the measured local subject overlaps keys 32% of the time."""
    r = check(human(seed, hold=140.0, gap=80.0, jitter=25.0))
    assert not r.flagged, (r.score, r.reasons)


def test_slow_hunt_and_peck_human_is_not_flagged():
    r = check(human(3, hold=200.0, gap=600.0, jitter=180.0))
    assert not r.flagged, (r.score, r.reasons)


def test_short_password_does_not_trip_the_variance_rules():
    """4 keys is too little evidence to call anyone mechanical."""
    s = human(2, codes=CODES, jitter=1.0)
    assert "low_variance_dd" not in fired(check(s, password_length=4))


def test_enter_submit_needs_no_pointer():
    assert "click_without_pointer" not in fired(check(human(4)))


def test_human_click_with_movement_is_not_flagged():
    pts = [{"type": "move", "t": 1000 + 16 * i, "x": 300 + 4 * i, "y": 400 - 3 * i} for i in range(20)]
    pts.append({"type": "down", "t": 1330, "x": 380, "y": 343, "buttons": 1, "target": "login"})
    r = check(human(5, pointer=pts, meta={"submit_via": "click"}))
    assert not r.flagged, (r.score, r.reasons)
    assert not fired(r) & {"click_without_pointer", "click_without_movement", "pointer_teleport"}


def test_touch_tap_is_not_a_teleport():
    pts = [{"type": "down", "t": 1330, "x": 380, "y": 343, "pointer_type": "touch", "target": "login"}]
    env = human_env(ua="Mozilla/5.0 (Linux; Android 14) Chrome/140.0 Mobile Safari/537.36",
                    max_touch_points=5, plugins=0, has_window_chrome=True)
    r = check(human(6, env=env, pointer=pts, meta={"submit_via": "click"}))
    assert not r.flagged, (r.score, r.reasons)


def test_a_repeat_login_by_the_same_human_is_not_a_replay():
    priors = [vec(human(i)).values for i in range(8)]
    r = check(human(99), vector=vec(human(99)), priors=priors)
    assert "replay" not in fired(r), r.reasons


# ---------------------------------------------------------------- bots caught


def test_untrusted_events_hard_flag():
    d = make_sample(codes=LONG_CODES, jitter=35.0, seed=1, trusted=False, env=human_env())
    r = check(Sample.model_validate(d))
    assert r.flagged and "untrusted_keys" in fired(r)


def test_webdriver_hard_flags():
    r = check(human(1, env=human_env(webdriver=True)))
    assert r.flagged and "webdriver" in fired(r)


def test_headless_ua_hard_flags():
    ua = "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/140.0.0.0 Safari/537.36"
    assert "headless_ua" in fired(check(human(1, env=human_env(ua=ua))))
    assert check(human(1, env=human_env(ua_brands=["HeadlessChrome", "Chromium"]))).flagged


def test_automation_globals_hard_flag():
    r = check(human(1, env=human_env(automation_globals=["cdc_adoQpoasnfa76pfcZLmcfl_Array"])))
    assert r.flagged and "automation_globals" in fired(r)


def test_weak_headless_tells_accumulate():
    """No single one blocks; a headless stack trips several at once."""
    env = human_env(ua="Mozilla/5.0 (X11; Linux x86_64) Chrome/140.0.0.0 Safari/537.36",
                    plugins=0, languages=[], outer_width=0, outer_height=0,
                    webgl_renderer="Google SwiftShader", has_window_chrome=False)
    single = check(human(1, env=human_env(webgl_renderer="Google SwiftShader")))
    assert not single.flagged, single.reasons  # software rendering alone must not block
    assert check(human(1, env=env)).flagged


def test_no_probe_alone_does_not_block_but_helps():
    r = check(human(1, env={}))
    assert "no_probe" in fired(r) and not r.flagged


def test_password_set_without_typing():
    s = human(1, codes=["KeyT"])
    r = check(s, password_length=10)
    assert r.flagged and "too_few_keys" in fired(r)
    empty = Sample.model_validate({"keystrokes": [], "pointer": [], "env": human_env(), "meta": {}})
    assert "no_typing" in fired(check(empty, password_length=10))


def test_shift_keys_do_not_count_against_the_length():
    """Typing "Roanl" presses Shift too; a 10-char password typed with 2 Shifts has
    12 keydowns and must still pass the count rule."""
    codes = ["ShiftLeft", "Period", "KeyT", "KeyI", "KeyE", "Digit5", "ShiftLeft",
             "KeyR", "KeyO", "KeyA", "KeyN", "KeyL"]
    assert not (fired(check(human(1, codes=codes), password_length=10)) & {"too_few_keys", "no_typing"})


def test_paste_is_reported_weakly_not_as_a_hard_block():
    s = human(1, codes=["KeyT"], meta={"submit_via": "enter", "had_paste": True})
    r = check(s, password_length=10)
    assert "paste" in fired(r) and "too_few_keys" not in fired(r)
    assert not r.flagged  # a human pasting from a password manager is not a bot


def typed(codes, holds, gaps, **over) -> Sample:
    """Exact control over timing, where make_sample clamps holds to >= 20 ms."""
    events, t = [], 500.0
    for c, h, g in zip(codes, holds, gaps):
        events += [{"code": c, "type": "down", "t": t, "field": "password", "trusted": True},
                   {"code": c, "type": "up", "t": t + h, "field": "password", "trusted": True}]
        t += g
    d = {"keystrokes": events, "pointer": [], "env": human_env(), "meta": {}}
    d.update(over)
    return Sample.model_validate(d)


def test_zero_hold_hard_flags():
    """Playwright/Puppeteer type(): keydown and keyup back to back."""
    rng = random.Random(3)
    n = len(LONG_CODES)
    r = check(typed(LONG_CODES, [rng.uniform(0, 2) for _ in range(n)],
                    [rng.gauss(120, 30) for _ in range(n)]))
    assert r.flagged and "impossible_hold" in fired(r)


def test_fixed_delay_bot_hard_flags():
    r = check(human(1, hold=60.0, gap=100.0, jitter=0.0))
    assert r.flagged
    assert fired(r) & {"identical_dd", "identical_hold", "low_variance_dd"}


def test_near_fixed_delay_bot_with_small_jitter_hard_flags():
    """A real browser driven by a script: 1-3 ms of event-loop jitter on a fixed delay."""
    r = check(human(1, hold=60.0, gap=100.0, jitter=1.5))
    assert r.flagged, (r.score, r.reasons)
    assert fired(r) & {"low_variance_dd", "low_variance_hold"}


def test_all_keys_pressed_at_once():
    rng = random.Random(0)
    events, t = [], 500.0
    for c in LONG_CODES:  # 2 ms apart: a script stuffing the field
        events += [{"code": c, "type": "down", "t": t, "field": "password", "trusted": True},
                   {"code": c, "type": "up", "t": t + 30 + rng.random(), "field": "password", "trusted": True}]
        t += 2.0
    r = check(Sample.model_validate({"keystrokes": events, "env": human_env(), "meta": {}}))
    assert r.flagged and "impossible_dd" in fired(r)


def test_round_number_timings_are_a_weak_tell():
    s = human(1, hold=100.0, gap=200.0, jitter=0.0)
    assert "round_timings" in fired(check(s))


def test_exact_replay_is_caught():
    s = human(7)
    v = vec(s)
    r = check(s, vector=v, priors=[[x + 500.0 for x in v.values], v.values])
    assert r.flagged and "replay" in fired(r)


def test_replay_with_small_added_jitter_is_still_caught():
    rng = random.Random(4)
    s = human(7)
    v = vec(s)
    noisy = [x + rng.uniform(-3, 3) for x in v.values]  # attacker fuzzes the timings a little
    r = check(s, vector=FeatureVector(names=v.names, values=noisy), priors=[v.values])
    assert r.flagged and "replay" in fired(r)


def test_heavily_perturbed_replay_escapes_the_replay_rule():
    """Honest limit: fuzz by +/- 25 ms and it is no longer a replay by this metric.
    It then has to survive the keystroke model instead, which is the point of fusion."""
    rng = random.Random(5)
    v = vec(human(7))
    noisy = [x + rng.uniform(-25, 25) for x in v.values]
    r = check(human(7), vector=FeatureVector(names=v.names, values=noisy), priors=[v.values])
    assert "replay" not in fired(r)


def test_click_with_no_pointer_events():
    r = check(human(1, meta={"submit_via": "click"}))
    assert "click_without_pointer" in fired(r)


def test_untrusted_pointer_hard_flags():
    pts = [{"type": "move", "t": 1000, "x": 10, "y": 10, "trusted": False},
           {"type": "down", "t": 1100, "x": 380, "y": 343, "trusted": False, "target": "login"}]
    r = check(human(1, pointer=pts, meta={"submit_via": "click"}))
    assert r.flagged and "untrusted_pointer" in fired(r)


def test_pointer_teleport_hard_flags():
    pts = [{"type": "move", "t": 1000, "x": 10, "y": 10},
           {"type": "move", "t": 1016, "x": 14, "y": 12},
           {"type": "down", "t": 1100, "x": 700, "y": 500, "buttons": 1, "target": "login"}]
    r = check(human(1, pointer=pts, meta={"submit_via": "click"}))
    assert r.flagged and "pointer_teleport" in fired(r)


def test_result_shape_and_explanations():
    r = check(human(1, env=human_env(webdriver=True)))
    assert r.name == "bot" and r.available
    assert r.contributions and r.contributions[0].feature.startswith("bot.")
    assert all(isinstance(x, str) and x for x in r.reasons)
    assert r.threshold == bot.THRESHOLD


def test_a_broken_sample_does_not_crash_the_check():
    s = human(1)
    s.env = {"languages": "not-a-list", "plugins": "many", "automation_globals": "cdc_"}
    assert isinstance(check(s).score, float)
