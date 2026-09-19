"""engine/keypad.py: the scrambled-keypad captcha. OWNER: Agent L (keypad).

Everything here runs on synthetic captchas from helpers.make_run: one parameter
set is one "person", two seeds of it are two captchas by that person, and a
different interval/hold is somebody else. **These numbers are synthetic** — they
show that the features separate two generators and that nothing blows up, not
that the keypad separates two humans. Only live runs can say that.
"""

from __future__ import annotations

import math

import pytest
from helpers import keypad_cells, make_run

from contracts import KEYPAD_BACK, KeypadRun
from engine import keypad, scorer
from test_bot import human_env

ENROLL_SEEDS = range(6)
NO_REPEATS = [0, 1, 2, 3, 4, 0]  # a target whose consecutive digits always differ


def run(**kw) -> KeypadRun:
    return KeypadRun.model_validate(make_run(**kw))


def impostor(seed: int, **kw) -> KeypadRun:
    """Clearly a different person: twice as slow to find each digit, heavier press."""
    return run(seed=seed, interval_ms=1040.0, interval_sd=140.0, hold_ms=175.0, **kw)


@pytest.fixture(scope="module")
def profile() -> keypad.Profile:
    return keypad.fit_profile([run(seed=i) for i in ENROLL_SEEDS])


def fired(result) -> set[str]:
    return {c.feature.removeprefix("bot.") for c in result.contributions}


# ---------------------------------------------------------------- vector shape


@pytest.mark.parametrize("device", ["mouse", "touch"])
def test_cognitive_vectors_have_the_declared_names_and_are_finite(device):
    cog, _ = keypad.tap_vectors(run(seed=3, device=device))
    assert cog, "a solved captcha must yield taps"
    for v in cog:
        assert v.names == keypad.NAMES_COG
        assert len(v.values) == len(keypad.NAMES_COG)
        assert all(math.isfinite(x) for x in v.values)


def test_mouse_run_yields_mouse_motor_vectors():
    _, motor = keypad.tap_vectors(run(seed=3, target=NO_REPEATS))
    assert motor
    for v in motor:
        assert v.names == keypad.NAMES_MOTOR["mouse"]
        assert len(v.values) == len(keypad.NAMES_MOTOR["mouse"])
        assert all(math.isfinite(x) for x in v.values)


def test_touch_run_yields_only_the_touch_motor_features():
    cog, motor = keypad.tap_vectors(run(seed=3, device="touch"))
    assert [v.names for v in motor] == [keypad.NAMES_MOTOR["touch"]] * len(motor)
    assert len(motor) == len(cog)  # a tap has no reach to fail: every tap lands
    assert all(math.isfinite(x) for v in motor for x in v.values)


def test_backspace_taps_are_not_samples_but_still_stop_the_clock():
    r = run(seed=5, errors=2)
    assert sum(1 for t in r.taps if t.digit == KEYPAD_BACK) == 2
    cog, _ = keypad.tap_vectors(r)
    assert len(cog) == sum(1 for t in r.taps if t.digit >= 0)

    # The tap after a backspace is timed from the backspace's release, not from
    # the digit tap before it.
    i = next(i for i, t in enumerate(r.taps) if t.digit == KEYPAD_BACK)
    after = r.taps[i + 1]
    digits_before = sum(1 for t in r.taps[:i + 1] if t.digit >= 0)
    got = cog[digits_before].values[keypad.NAMES_COG.index("kp.interval_ms")]
    assert got == pytest.approx(after.t_down - r.taps[i].t_up, abs=0.01)


def test_a_repeated_digit_has_no_reach_so_no_motor_vector():
    r = run(seed=7, target=[2, 2, 3, 3, 1, 1])
    cog, motor = keypad.tap_vectors(r)
    assert len(cog) == 6
    assert len(motor) == 3, "the three repeats start on the key they are aiming at"


# ---------------------------------------------------------------- device class


def test_device_class_follows_the_taps():
    assert keypad.device_class(run(seed=1)) == "mouse"
    assert keypad.device_class(run(seed=1, device="touch")) == "touch"
    empty = run(seed=1)
    empty.taps = []
    assert keypad.device_class(empty) == "unknown"


def test_an_unrecognised_pointer_type_gets_no_motor_vectors():
    r = run(seed=1)
    for t in r.taps:
        t.pointer_type = "pen"
    assert keypad.device_class(r) == "unknown"
    cog, motor = keypad.tap_vectors(r)
    assert cog and motor == []


# ---------------------------------------------------------------- profile


def test_fit_profile_uses_a_run_level_threshold(profile):
    assert scorer.THRESHOLD_FLOOR <= profile.cog.threshold <= scorer.THRESHOLD_CEILING
    assert 1.0 <= profile.cog.threshold <= 1.9
    assert len(profile.cog.loo) == len(ENROLL_SEEDS), "one leave-one-out score per captcha, not per tap"
    assert profile.cog.n == 6 * 6  # six captchas of six digits
    assert profile.device_class == "mouse"
    assert profile.n_runs == 6
    assert profile.motor is not None


def test_profile_round_trips_through_a_dict(profile):
    back = keypad.Profile.from_dict(profile.to_dict())
    assert back.cog.names == profile.cog.names
    assert back.cog.center == profile.cog.center
    assert back.cog.threshold == profile.cog.threshold
    assert back.cog.loo == profile.cog.loo
    assert back.motor is not None and back.motor.names == profile.motor.names
    assert back.device_class == profile.device_class and back.n_runs == profile.n_runs
    assert keypad.score_runs(back, [run(seed=101)])[0].score == \
        keypad.score_runs(profile, [run(seed=101)])[0].score


def test_no_motor_model_from_too_few_runs():
    p = keypad.fit_profile([run(seed=i) for i in range(2)])
    assert p.motor is None
    assert p.cog.n == 12  # the cognitive half still fits


def test_motor_model_only_sees_the_majority_device_class():
    p = keypad.fit_profile([run(seed=i) for i in range(4)] + [run(seed=9, device="touch")])
    assert p.device_class == "mouse"
    assert p.motor is not None and p.motor.names == keypad.NAMES_MOTOR["mouse"]


# ---------------------------------------------------------------- scoring


def test_the_same_person_passes(profile):
    cog, motor = keypad.score_runs(profile, [run(seed=101), run(seed=102)])
    assert cog.available and not cog.flagged
    assert cog.score < profile.cog.threshold * 0.95
    assert motor.available and not motor.flagged


def test_a_different_person_is_flagged(profile):
    cog, motor = keypad.score_runs(profile, [impostor(201), impostor(202)])
    assert cog.flagged and cog.score > profile.cog.threshold * 2
    assert motor.flagged and all(r.endswith("(advisory)") for r in motor.reasons)


def test_a_faster_impostor_is_flagged_too(profile):
    cog, _ = keypad.score_runs(profile, [run(seed=501, interval_ms=260.0, interval_sd=40.0, hold_ms=45.0),
                                         run(seed=502, interval_ms=260.0, interval_sd=40.0, hold_ms=45.0)])
    assert cog.flagged and cog.score > profile.cog.threshold * 1.5


def test_the_explanation_is_a_sentence_a_judge_can_read(profile):
    cog, _ = keypad.score_runs(profile, [impostor(201), impostor(202)])
    assert cog.reasons and all(" " in r for r in cog.reasons)
    assert any("than usual" in r for r in cog.reasons)
    assert not any(n in r for r in cog.reasons for n in keypad.NAMES_COG), \
        "no feature names in text shown to a human"
    assert 0 < len(cog.contributions) <= scorer.MAX_CONTRIBUTIONS
    assert sum(c.deviation for c in cog.contributions) == pytest.approx(cog.score, abs=1e-9)


def test_a_captcha_with_no_taps_is_not_scorable(profile):
    empty = run(seed=1)
    empty.taps = []
    cog, motor = keypad.score_runs(profile, [empty])
    assert not cog.available and not cog.flagged
    assert not motor.available


# ---------------------------------------------------------------- cross-device


def test_the_same_person_on_a_phone_passes_and_movement_is_withheld(profile):
    cog, motor = keypad.score_runs(profile, [run(seed=301, device="touch"), run(seed=302, device="touch")])
    assert not cog.flagged, "the cognitive half is the whole point: it must survive the device change"
    assert cog.score < profile.cog.threshold * 0.95
    assert not motor.available
    assert "not comparable" in motor.reasons[0] and "touch screen" in motor.reasons[0]


def test_an_impostor_on_a_phone_is_still_flagged(profile):
    cog, motor = keypad.score_runs(profile, [impostor(401, device="touch"), impostor(402, device="touch")])
    assert cog.flagged
    assert not motor.available  # a class change never produces a motor verdict, either way


def test_a_touch_profile_scores_touch_runs(profile):
    p = keypad.fit_profile([run(seed=i, device="touch") for i in ENROLL_SEEDS])
    assert p.device_class == "touch" and p.motor is not None
    assert p.motor.names == keypad.NAMES_MOTOR["touch"]
    cog, motor = keypad.score_runs(p, [run(seed=301, device="touch")])
    assert not cog.flagged and motor.available
    assert keypad.score_runs(p, [impostor(401, device="touch")])[0].flagged


# ---------------------------------------------------------------- run_stats


def test_run_stats_counts_what_the_dashboard_shows():
    r = run(seed=11, errors=2)
    s = keypad.run_stats(r)
    assert s["errors"] == 2 and s["backspaces"] == 2
    assert s["taps"] == 8  # six digits plus the two wrong ones; backspaces are not taps of a digit
    assert s["completed"] is True
    assert s["first_tap_ms"] == pytest.approx(r.taps[0].t_down - r.shown_at)
    assert s["duration_ms"] == pytest.approx(r.taps[-1].t_up - r.shown_at)
    assert s["device_class"] == "mouse"
    assert keypad.run_stats(run(seed=11))["errors"] == 0


# ---------------------------------------------------------------- bot rules


@pytest.mark.parametrize("device", ["mouse", "touch"])
def test_no_bot_rule_fires_on_a_real_captcha(device):
    r = keypad.bot_check([run(seed=21, device=device)])
    assert fired(r) == set() and not r.flagged and r.score == 0.0


def test_a_batch_of_clean_captchas_is_not_flagged():
    r = keypad.bot_check([run(seed=22), run(seed=23, errors=1)])
    assert not r.flagged, r.reasons


def test_untrusted_events_are_a_strong_tell():
    r = keypad.bot_check([run(seed=24, trusted=False)])
    assert "keypad_untrusted_taps" in fired(r) and "keypad_untrusted_pointer" in fired(r)
    assert r.flagged


def test_the_environment_probe_still_applies():
    r = keypad.bot_check([run(seed=24, env=human_env(webdriver=True))])
    assert "webdriver" in fired(r) and r.flagged


def test_a_new_layout_cannot_be_searched_in_under_120_ms():
    r = keypad.bot_check([run(seed=25, target=NO_REPEATS, interval_ms=100.0, interval_sd=8.0,
                              first_extra_ms=400.0)])
    assert "keypad_impossible_search" in fired(r) and r.flagged


def test_a_repeated_digit_is_allowed_to_be_fast():
    """The same key twice needs no new search, so the fast-tap rule must exempt it."""
    d = make_run(seed=26, target=[3, 3, 3, 3, 3, 3], interval_ms=100.0, interval_sd=8.0,
                 first_extra_ms=400.0)
    assert "keypad_impossible_search" not in fired(keypad.bot_check([KeypadRun.model_validate(d)]))


def test_tapping_before_the_digits_could_be_read_is_a_weak_tell():
    d = make_run(seed=27)
    d["shown_at"] = d["taps"][0]["t_down"] - 100.0
    r = keypad.bot_check([KeypadRun.model_validate(d)])
    assert "keypad_instant_start" in fired(r)


def test_identical_gaps_between_taps_are_mechanical():
    d = make_run(seed=28)
    t = 0.0
    for tap in d["taps"]:
        tap["t_down"] = t + 300.0
        tap["t_up"] = tap["t_down"] + 50.0
        t = tap["t_up"]
    r = keypad.bot_check([KeypadRun.model_validate(d)])
    assert "keypad_identical_intervals" in fired(r) and r.flagged


def test_impossibly_short_presses_are_mechanical():
    d = make_run(seed=29)
    for tap in d["taps"][:3]:
        tap["t_up"] = tap["t_down"] + 3.0
    r = keypad.bot_check([KeypadRun.model_validate(d)])
    assert "keypad_impossible_hold" in fired(r) and r.flagged


def test_pixel_perfect_landings_are_a_script():
    d = make_run(seed=30)
    cells = keypad_cells()
    for tap in d["taps"][:4]:
        rx, ry, rw, rh = cells["back" if tap["cell"] < 0 else str(tap["cell"])]
        tap["x"], tap["y"] = rx + rw / 2, ry + rh / 2
    r = keypad.bot_check([KeypadRun.model_validate(d)])
    assert "keypad_pixel_perfect" in fired(r) and r.flagged


def test_a_mouse_that_never_moved_is_a_script():
    d = make_run(seed=31)
    d["pointer"] = [e for e in d["pointer"] if e["type"] != "move"]
    r = keypad.bot_check([KeypadRun.model_validate(d)])
    assert "keypad_no_movement" in fired(r) and r.flagged


@pytest.mark.parametrize("device, env, rule", [
    ("touch", None, "keypad_touch_without_touchscreen"),   # touch taps, desktop env
    ("mouse", {"ua_mobile": True}, "keypad_mouse_on_phone"),
])
def test_the_device_contradicting_itself_is_a_weak_tell(device, env, rule):
    d = make_run(seed=32, device=device, env=human_env(**(env or {})))
    assert rule in fired(keypad.bot_check([KeypadRun.model_validate(d)]))


def test_an_unsolved_captcha_is_a_weak_tell():
    d = make_run(seed=33)
    d["completed"] = False
    assert "keypad_not_solved" in fired(keypad.bot_check([KeypadRun.model_validate(d)]))
    d = make_run(seed=33)
    d["taps"] = d["taps"][:-1]  # the last digit was never entered
    assert "keypad_not_solved" in fired(keypad.bot_check([KeypadRun.model_validate(d)]))


def test_no_runs_at_all_is_reported_rather_than_crashing():
    r = keypad.bot_check([])
    assert "keypad_no_runs" in fired(r) and not r.flagged


def test_a_malformed_run_does_not_crash_the_check():
    r = run(seed=34)
    r.cells = {}  # rects lost: the landing rule cannot run
    result = keypad.bot_check([r])
    assert isinstance(result.score, float)
