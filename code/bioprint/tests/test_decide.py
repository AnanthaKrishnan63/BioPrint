"""Routing in engine/decide.py: the device x keystroke table, extended with the
keypad routes (see research/big_idea/15 Scrambled Keypad.md)."""

from contracts import SignalResult
from engine.decide import BORDERLINE, decide, decide_keypad


def sig(name, *, available=True, score=0.0, threshold=1.0, flagged=False, reasons=()):
    return SignalResult(name=name, available=available, score=score, threshold=threshold,
                        flagged=flagged, reasons=list(reasons))


def ks(score, threshold=1.9):
    return sig("keystroke", score=score, threshold=threshold, flagged=score > threshold)


SAME = sig("device", score=0.0, threshold=6.0)
NEW = sig("device", score=30.0, threshold=6.0, flagged=True, reasons=["different screen"])
BOT = sig("bot", score=2.0, threshold=0.75, flagged=True, reasons=["webdriver"])
HUMAN = sig("bot", score=0.0, threshold=0.75)


def test_bot_blocks_first():
    assert decide([BOT, SAME, ks(0.5)])[0] == "block"
    assert decide([BOT, NEW, ks(0.5)], keypad_enrolled=True, new_class=True)[0] == "block"


def test_same_device_allow_and_block():
    assert decide([HUMAN, SAME, ks(1.0)])[0] == "allow"
    assert decide([HUMAN, SAME, ks(2.5)])[0] == "block"


def test_new_device_bad_rhythm_blocks_whatever_the_keypad():
    assert decide([HUMAN, NEW, ks(2.5)], keypad_enrolled=True, new_class=True)[0] == "block"


def test_new_device_without_keypad_profile_is_typing_step_up():
    assert decide([HUMAN, NEW, ks(1.0)])[0] == "step_up"
    assert decide([HUMAN, NEW, ks(1.8)])[0] == "step_up"  # borderline, but no keypad to go to


def test_new_device_same_class_comfortable_pass_is_typing_step_up():
    assert decide([HUMAN, NEW, ks(1.0)], keypad_enrolled=True)[0] == "step_up"


def test_new_class_routes_to_keypad():
    d, reasons = decide([HUMAN, NEW, ks(1.0)], keypad_enrolled=True, new_class=True)
    assert d == "keypad"
    assert any("different kind of device" in r for r in reasons)


def test_no_key_timing_routes_to_keypad():
    d, reasons = decide([NEW, HUMAN, sig("keystroke", available=False)], keypad_enrolled=True)
    assert d == "keypad"
    assert any("no key timing" in r for r in reasons)


def test_borderline_routes_to_keypad():
    thr = 1.9
    just_under = BORDERLINE * thr - 0.01
    assert decide([HUMAN, NEW, ks(just_under, thr)], keypad_enrolled=True)[0] == "step_up"
    d, reasons = decide([HUMAN, NEW, ks(BORDERLINE * thr + 0.01, thr)], keypad_enrolled=True)
    assert d == "keypad"
    assert any("only just" in r for r in reasons)


def test_after_typing_step_up():
    # Median comfortably under the limit: allow, new device advisory.
    d, reasons = decide([HUMAN, NEW, ks(1.0)], after_step_up=True, keypad_enrolled=True)
    assert d == "allow" and any("advisory" in r for r in reasons)
    # Median still near the limit: the keypad settles it.
    assert decide([HUMAN, NEW, ks(1.7)], after_step_up=True, keypad_enrolled=True)[0] == "keypad"
    # ... unless there is no keypad profile.
    assert decide([HUMAN, NEW, ks(1.7)], after_step_up=True)[0] == "allow"
    assert decide([HUMAN, NEW, ks(2.5)], after_step_up=True, keypad_enrolled=True)[0] == "block"


def test_nothing_to_compare_never_allows():
    assert decide([HUMAN, SAME, sig("keystroke", available=False)])[0] == "block"


def test_pointer_is_advisory():
    d, reasons = decide([HUMAN, SAME, ks(1.0), sig("pointer", score=3.0, flagged=True)])
    assert d == "allow" and any("pointer" in r for r in reasons)


def test_decide_keypad():
    kp_ok = sig("keypad", score=0.8, threshold=1.5)
    kp_bad = sig("keypad", score=2.0, threshold=1.5, flagged=True, reasons=["slower search"])
    motor_off = sig("keypad_motor", available=False, reasons=["touch vs mouse"])
    motor_bad = sig("keypad_motor", score=3.0, threshold=1.5, flagged=True)
    assert decide_keypad([BOT, NEW, kp_ok])[0] == "block"
    assert decide_keypad([HUMAN, NEW, kp_bad, motor_off])[0] == "block"
    d, reasons = decide_keypad([HUMAN, NEW, kp_ok, motor_off])
    assert d == "allow" and any("keypad behaviour alone" in r for r in reasons)
    d, reasons = decide_keypad([HUMAN, NEW, kp_ok, motor_bad])
    assert d == "allow" and any("advisory" in r for r in reasons)
    assert decide_keypad([HUMAN, NEW, sig("keypad", available=False)])[0] == "block"
    assert decide_keypad([HUMAN, NEW])[0] == "block"
