"""Pairing and vector layout. OWNER: Agent A.

The cases here are the ones a real browser stream throws at us (phantom keydowns,
orphan keyups, rollover, a keyup that lands in another field), not invented ones:
each references the measurement in research/big_idea/12 Measurements.md.
"""

from contracts import Sample
from engine import features as F


def sample(*events, field="password") -> Sample:
    """`("KeyT", "down", 100)` tuples -> a Sample. A 4th item overrides the field."""
    ks = [{"code": c, "type": t, "t": float(ms), "field": (e[3] if len(e) > 3 else field)}
          for e in events for c, t, ms in [e[:3]]]
    return Sample.model_validate({"keystrokes": ks})


def codes(ks):
    return [k.code for k in ks]


# ---------------------------------------------------------------- pairing

def test_plain_pairing():
    ks = F.password_keystrokes(sample(("KeyT", "down", 100), ("KeyT", "up", 190),
                                      ("KeyI", "down", 250), ("KeyI", "up", 330)))
    assert codes(ks) == ["KeyT", "KeyI"]
    assert [(k.down, k.up) for k in ks] == [(100, 190), (250, 330)]


def test_phantom_duplicate_keydown_uses_the_later_press():
    # Firefox fires the phantom ~5 ms BEFORE the genuine press (12 Measurements, S3).
    # The real press is the second one, so the dwell is 190-205, not 190-200.
    ks = F.password_keystrokes(sample(("KeyG", "down", 200), ("KeyG", "down", 205), ("KeyG", "up", 395)))
    assert len(ks) == 1
    assert (ks[0].down, ks[0].up) == (205, 395)


def test_auto_repeat_keeps_the_first_press():
    # Far outside the phantom window: this is a held key the client failed to
    # filter with event.repeat. The press that matters is the first.
    ks = F.password_keystrokes(sample(("KeyG", "down", 200), ("KeyG", "down", 700), ("KeyG", "up", 800)))
    assert len(ks) == 1 and ks[0].down == 200


def test_keyup_without_keydown_is_ignored():
    ks = F.password_keystrokes(sample(("KeyT", "up", 50), ("KeyI", "down", 100), ("KeyI", "up", 180)))
    assert codes(ks) == ["KeyI"]


def test_unreleased_key_survives_as_a_lower_bound():
    # Submitting with Enter serialises the sample while the last key is still down.
    # Dropping it would reject the sample of every fast typist, forever.
    ks = F.password_keystrokes(sample(("KeyT", "down", 100), ("KeyT", "up", 190), ("KeyI", "down", 250),
                                      ("Enter", "down", 300)))
    assert codes(ks) == ["KeyT", "KeyI"]
    assert ks[1].up == 300  # closed at the last event seen, never before its own press


def test_keyup_in_another_field_still_counts():
    # Tab/Enter moves focus between keydown and keyup; the press was still typed
    # into the password field.
    s = sample(("KeyT", "down", 100, "password"), ("KeyT", "up", 190, "username"))
    assert codes(F.password_keystrokes(s)) == ["KeyT"]


def test_username_keys_are_not_password_keys():
    s = sample(("KeyA", "down", 10, "username"), ("KeyA", "up", 60, "username"),
               ("KeyT", "down", 100, "password"), ("KeyT", "up", 190, "password"))
    assert codes(F.password_keystrokes(s)) == ["KeyT"]


def test_overlapping_keys_are_ordered_by_keydown():
    # Rollover: T is released only after I and E are already down. Measured at 32%
    # of this subject's keystrokes, so it is the normal case, not an edge case.
    ks = F.password_keystrokes(sample(("KeyT", "down", 100), ("KeyI", "down", 150), ("KeyE", "down", 200),
                                      ("KeyI", "up", 230), ("KeyT", "up", 240), ("KeyE", "up", 280)))
    assert codes(ks) == ["KeyT", "KeyI", "KeyE"]
    v = F.keystroke_vector(ks)
    assert dict(zip(v.names, v.values))["UD.KeyT#0.KeyI#1"] == -90  # negative = overlap


def test_same_key_twice_pairs_in_order():
    ks = F.password_keystrokes(sample(("KeyA", "down", 100), ("KeyA", "up", 180),
                                      ("KeyA", "down", 260), ("KeyA", "up", 330)))
    assert [(k.down, k.up) for k in ks] == [(100, 180), (260, 330)]


def test_events_out_of_order_are_sorted():
    ks = F.password_keystrokes(sample(("KeyI", "up", 330), ("KeyT", "down", 100),
                                      ("KeyI", "down", 250), ("KeyT", "up", 190)))
    assert codes(ks) == ["KeyT", "KeyI"]


# ---------------------------------------------------------------- modifiers

def test_shift_hand_does_not_change_the_template():
    left = sample(("ShiftLeft", "down", 90), ("KeyR", "down", 120), ("KeyR", "up", 200),
                  ("ShiftLeft", "up", 210))
    right = sample(("ShiftRight", "down", 90), ("KeyR", "down", 120), ("KeyR", "up", 200),
                   ("ShiftRight", "up", 210))
    assert codes(F.password_keystrokes(left)) == codes(F.password_keystrokes(right)) == ["KeyR"]
    template = F.template_codes(F.keystroke_vector(F.password_keystrokes(left)))
    assert F.needs_retype(right, template) is None


def test_modifiers_are_canonicalised_even_when_kept():
    assert F.canonical_code("ShiftRight") == "Shift"
    assert F.canonical_code("ControlLeft") == "Control"
    assert F.canonical_code("OSRight") == "Meta"
    assert F.canonical_code("KeyT") == "KeyT"


def test_holding_shift_across_two_letters_still_matches():
    # One Shift press for "RO" on one login, two on the next: same template either
    # way, because modifiers are not part of the positional sequence.
    one = sample(("ShiftLeft", "down", 80), ("KeyR", "down", 100), ("KeyR", "up", 170),
                 ("KeyO", "down", 200), ("KeyO", "up", 260), ("ShiftLeft", "up", 270))
    two = sample(("ShiftLeft", "down", 80), ("KeyR", "down", 100), ("KeyR", "up", 170),
                 ("ShiftLeft", "up", 175), ("ShiftRight", "down", 190), ("KeyO", "down", 200),
                 ("KeyO", "up", 260), ("ShiftRight", "up", 270))
    t = F.template_codes(F.keystroke_vector(F.password_keystrokes(one)))
    assert t == ["KeyR", "KeyO"]
    assert F.needs_retype(two, t) is None


# ---------------------------------------------------------------- retype rules

def test_correction_and_navigation_force_a_retype():
    for bad in ("Backspace", "Delete", "ArrowLeft", "Home"):
        s = sample(("KeyT", "down", 100), ("KeyT", "up", 150), (bad, "down", 200), (bad, "up", 250))
        assert "again" in F.needs_retype(s)


def test_empty_password_field_is_unscorable():
    assert F.needs_retype(sample(("KeyT", "down", 1, "username"), ("KeyT", "up", 9, "username"))) is not None


def test_enter_and_tab_are_not_part_of_the_password():
    s = sample(("Tab", "down", 10), ("Tab", "up", 20), ("KeyT", "down", 100), ("KeyT", "up", 190),
               ("Enter", "down", 300), ("Enter", "up", 380))
    assert codes(F.password_keystrokes(s)) == ["KeyT"]


def test_different_key_sequence_is_rejected():
    s = sample(("KeyT", "down", 100), ("KeyT", "up", 190))
    assert F.needs_retype(s, ["KeyT", "KeyI"]) is not None
    assert F.needs_retype(s, ["KeyT"]) is None


# ---------------------------------------------------------------- vector layout

def test_vector_layout_matches_cmu():
    ks = F.password_keystrokes(sample(("KeyT", "down", 100), ("KeyT", "up", 190),
                                      ("KeyI", "down", 250), ("KeyI", "up", 330)))
    v = F.keystroke_vector(ks)
    assert v.names == ["H.KeyT#0", "DD.KeyT#0.KeyI#1", "UD.KeyT#0.KeyI#1", "H.KeyI#1"]
    assert v.values == [90, 150, 60, 80]
    assert len(v.names) == 3 * len(ks) - 2 == len(v.values)


def test_template_codes_round_trip_with_repeated_keys():
    ks = F.password_keystrokes(sample(("KeyA", "down", 100), ("KeyA", "up", 150),
                                      ("KeyB", "down", 200), ("KeyB", "up", 250),
                                      ("KeyA", "down", 300), ("KeyA", "up", 350)))
    assert F.template_codes(F.keystroke_vector(ks)) == ["KeyA", "KeyB", "KeyA"]


def test_vector_is_empty_for_one_key_but_still_valid():
    v = F.keystroke_vector(F.password_keystrokes(sample(("KeyT", "down", 100), ("KeyT", "up", 190))))
    assert v.names == ["H.KeyT#0"] and v.values == [90]
