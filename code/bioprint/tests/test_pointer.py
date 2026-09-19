"""Pointer feature tests. OWNER: Agent C (pointer).

Everything here runs on synthetic trajectories generated below: a minimum-jerk
primary sub-movement to an aim point, optional overshoot plus a corrective
sub-movement, a lateral arc, a hover, and a click of a given hold time. A
"person" is one parameter set; sampling it twice with different seeds gives two
tries by the same person.
"""

from __future__ import annotations

import math
import random

import pytest
from helpers import make_sample

from contracts import Sample
from engine import pointer, scorer

RECT = [100.0, 300.0, 200.0, 40.0]  # the submit button, [x, y, w, h]
CENTRE = (RECT[0] + RECT[2] / 2, RECT[1] + RECT[3] / 2)


def _min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def make_pointer_sample(
    *,
    start=(140.0, 250.0),
    rect=RECT,
    ms_per_bit=180.0,  # Fitts slope: movement time = ms_per_bit * log2(D/W + 1)
    curve=0.10,  # lateral arc, as a fraction of the straight-line distance
    overshoot=0.05,  # primary sub-movement aims this far past the landing point
    corrections=1,  # corrective sub-movements after the primary one
    land=(0.0, 0.0),  # landing offset from the centre, in button widths/heights
    hover_ms=120.0,  # motionless on the button before pressing
    hold_ms=90.0,  # press -> release
    reaction_ms=350.0,  # last key released -> pointer starts moving
    step_ms=8.0,  # sampling interval of the captured moves
    jitter=0.6,  # px of hand/sensor noise
    wander=True,  # some unrelated movement long before the approach
    submit_via="click",
    seed=0,
) -> dict:
    rng = random.Random(seed)
    s = make_sample(seed=seed)  # keystrokes; ends with a keyup
    t_key = max(k["t"] for k in s["keystrokes"] if k["type"] == "up")

    rx, ry, rw, rh = rect
    cx, cy = rx + rw / 2, ry + rh / 2
    landing = (cx + land[0] * rw, cy + land[1] * rh)
    dx, dy = landing[0] - start[0], landing[1] - start[1]
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux  # unit normal, for the arc

    # Fitts index of difficulty against the button's extent along the approach.
    width = min(rw / abs(ux) if abs(ux) > 1e-9 else math.inf, rh / abs(uy) if abs(uy) > 1e-9 else math.inf)
    bits = math.log2(dist / width + 1.0)
    primary_ms = ms_per_bit * bits

    events = []
    t = 0.0
    if wander:
        for i in range(12):  # an earlier, unrelated episode, separated by a long pause
            events.append({"type": "move", "t": 40.0 + 10.0 * i, "x": 600.0 + 3 * i, "y": 120.0 + 2 * i,
                           "pointer_type": "mouse", "buttons": 0, "target": None, "trusted": True})

    def move(t, x, y):
        events.append({"type": "move", "t": t, "x": round(x + rng.gauss(0, jitter), 1),
                       "y": round(y + rng.gauss(0, jitter), 1), "pointer_type": "mouse",
                       "buttons": 0, "target": None, "trusted": True})

    t = t_key + reaction_ms
    # Primary sub-movement, aiming `overshoot` of the distance past the landing point.
    aim = (start[0] + ux * dist * (1 + overshoot), start[1] + uy * dist * (1 + overshoot))
    n = max(3, int(primary_ms / step_ms))
    for i in range(1, n + 1):
        tau = i / n
        f = _min_jerk(tau)
        x = start[0] + (aim[0] - start[0]) * f + px * curve * dist * math.sin(math.pi * tau)
        y = start[1] + (aim[1] - start[1]) * f + py * curve * dist * math.sin(math.pi * tau)
        move(t + tau * primary_ms, x, y)
    t += primary_ms
    here = aim
    # Corrective sub-movements: each closes half the remaining error, slowly.
    for c in range(corrections):
        corr_ms = 90.0 + 30.0 * c
        goal = (here[0] + (landing[0] - here[0]) * (1.0 if c == corrections - 1 else 0.5),
                here[1] + (landing[1] - here[1]) * (1.0 if c == corrections - 1 else 0.5))
        m = max(3, int(corr_ms / step_ms))
        for i in range(1, m + 1):
            tau = i / m
            f = _min_jerk(tau)
            move(t + tau * corr_ms, here[0] + (goal[0] - here[0]) * f, here[1] + (goal[1] - here[1]) * f)
        t += corr_ms
        here = goal
    if corrections == 0:
        here = landing

    t_down = t + hover_ms
    for kind, tt in (("down", t_down), ("up", t_down + hold_ms)):
        events.append({"type": kind, "t": tt, "x": here[0], "y": here[1], "pointer_type": "mouse",
                       "buttons": 1 if kind == "down" else 0, "target": "login", "trusted": True})

    s["pointer"] = events
    s["meta"] = {"submit_via": submit_via, "targets": {"submit": list(rect)}}
    return s


def vec(**kw):
    v = pointer.pointer_vector(Sample.model_validate(make_pointer_sample(**kw)))
    assert v is not None
    return dict(zip(v.names, v.values))


# ------------------------------------------------------------------ shape


def test_vector_is_fixed_length_with_stable_names():
    a = pointer.pointer_vector(Sample.model_validate(make_pointer_sample(seed=1)))
    b = pointer.pointer_vector(Sample.model_validate(
        make_pointer_sample(seed=2, start=(900.0, 80.0), ms_per_bit=300, corrections=3,
                            curve=0.3, land=(0.3, -0.2), hover_ms=800, hold_ms=210)))
    assert a.names == b.names == pointer.NAMES
    assert len(a.values) == len(b.values) == len(pointer.NAMES)
    assert all(math.isfinite(x) for x in a.values + b.values)


# ------------------------------------------------------------------ None cases


def test_none_when_submitted_with_enter():
    assert pointer.pointer_vector(Sample.model_validate(make_pointer_sample(submit_via="enter"))) is None


def test_none_without_pointer_data():
    assert pointer.pointer_vector(Sample.model_validate(make_sample())) is None


def test_none_when_no_press_landed_on_the_button():
    s = make_pointer_sample()
    for e in s["pointer"]:
        if e["type"] in ("down", "up"):
            e["x"], e["y"] = 10.0, 10.0  # clicked somewhere else entirely
    assert pointer.pointer_vector(Sample.model_validate(s)) is None


def test_none_when_the_approach_is_too_short():
    s = make_pointer_sample()
    s["pointer"] = [e for e in s["pointer"] if e["type"] != "move"][:2] + \
                   [e for e in s["pointer"] if e["type"] in ("down", "up")]
    assert pointer.pointer_vector(Sample.model_validate(s)) is None


def test_none_when_the_pointer_never_left_the_button():
    assert pointer.pointer_vector(Sample.model_validate(
        make_pointer_sample(start=(CENTRE[0] + 3, CENTRE[1] + 2), wander=False))) is None


def test_none_without_a_submit_rect():
    s = make_pointer_sample()
    s["meta"]["targets"] = {}
    assert pointer.pointer_vector(Sample.model_validate(s)) is None


# ------------------------------------------------------------------ plausibility


def test_values_are_plausible():
    f = vec(seed=3, hold_ms=95.0, hover_ms=150.0, reaction_ms=400.0, ms_per_bit=200.0)
    assert f["pointer.hold_ms"] == pytest.approx(95.0, abs=2)
    # arrival on the button happens during the primary movement here, so hover also
    # covers the one corrective sub-movement (90 ms); the motionless part is settle.
    assert f["pointer.settle_ms"] == pytest.approx(150.0, abs=20)
    assert f["pointer.hover_ms"] > f["pointer.settle_ms"]
    assert f["pointer.key_to_move_ms"] == pytest.approx(400.0, abs=20)
    # a single smooth movement recovers the generator's Fitts slope; corrective
    # sub-movements add time on top of it, as they do for a real hand.
    assert vec(seed=3, ms_per_bit=200.0, corrections=0, overshoot=0.0)["pointer.ms_per_bit"] == \
        pytest.approx(200.0, rel=0.1)
    assert f["pointer.ms_per_bit"] > 200.0
    assert 0.5 < f["pointer.path_efficiency"] <= 1.0
    assert 0.0 <= f["pointer.curvature"] < 0.3
    assert 1.0 < f["pointer.peak_mean_ratio"] < 4.0
    assert 0.1 < f["pointer.time_to_peak"] < 0.8
    assert f["pointer.speed_variation"] >= 1.0
    assert abs(f["pointer.land_dx"]) <= 0.5 and abs(f["pointer.land_dy"]) <= 0.5


def test_landing_offset_is_relative_to_the_button():
    f = vec(seed=4, land=(0.25, -0.2), jitter=0.0)
    assert f["pointer.land_dx"] == pytest.approx(0.25, abs=0.02)
    assert f["pointer.land_dy"] == pytest.approx(-0.2, abs=0.05)


def test_a_curved_wandering_path_is_less_efficient_and_more_curved():
    straight = vec(seed=5, curve=0.0, corrections=0, overshoot=0.0)
    curved = vec(seed=5, curve=0.25)
    assert curved["pointer.path_efficiency"] < straight["pointer.path_efficiency"]
    assert curved["pointer.curvature"] > straight["pointer.curvature"] + 0.1


def test_corrective_submovements_show_up_in_speed_variation_and_overshoot():
    clean = vec(seed=6, corrections=0, overshoot=0.0, curve=0.0)
    messy = vec(seed=6, corrections=2, overshoot=0.15, curve=0.0)
    assert messy["pointer.speed_variation"] > clean["pointer.speed_variation"] + 0.3
    assert messy["pointer.overshoot"] > clean["pointer.overshoot"] + 0.05
    assert messy["pointer.homing_frac"] > clean["pointer.homing_frac"]


# ------------------------------------------------------------------ invariances


def test_translating_the_whole_layout_changes_nothing():
    base = make_pointer_sample(seed=7, jitter=0.0)
    moved = make_pointer_sample(seed=7, jitter=0.0, start=(140.0 + 37, 250.0 - 21),
                                rect=[RECT[0] + 37, RECT[1] - 21, RECT[2], RECT[3]])
    a = pointer.pointer_vector(Sample.model_validate(base))
    b = pointer.pointer_vector(Sample.model_validate(moved))
    assert a.values == pytest.approx(b.values, rel=1e-6, abs=1e-6)


def test_shape_features_do_not_depend_on_where_the_cursor_started():
    near = vec(seed=8, start=(180.0, 260.0), jitter=0.0)
    far = vec(seed=8, start=(520.0, 60.0), jitter=0.0)
    for k in ("pointer.ms_per_bit", "pointer.path_efficiency", "pointer.curvature",
              "pointer.peak_mean_ratio", "pointer.time_to_peak", "pointer.overshoot"):
        assert near[k] == pytest.approx(far[k], rel=0.25, abs=0.05), k


def test_sampling_rate_does_not_move_the_features():
    fast = vec(seed=9, step_ms=4.0, jitter=0.0)
    slow = vec(seed=9, step_ms=16.0, jitter=0.0)
    for k in ("pointer.path_efficiency", "pointer.peak_mean_ratio", "pointer.time_to_peak",
              "pointer.ms_per_bit", "pointer.curvature"):
        assert fast[k] == pytest.approx(slow[k], rel=0.15, abs=0.05), k


def test_an_earlier_unrelated_episode_is_ignored():
    with_wander = vec(seed=10, wander=True, jitter=0.0)
    without = vec(seed=10, wander=False, jitter=0.0)
    assert with_wander == pytest.approx(without, rel=1e-6, abs=1e-6)


# ------------------------------------------------------------------ separation


ALICE = dict(ms_per_bit=170.0, curve=0.06, overshoot=0.03, corrections=1,
             land=(-0.10, 0.05), hover_ms=90.0, hold_ms=80.0, reaction_ms=320.0)
BOB = dict(ms_per_bit=300.0, curve=0.22, overshoot=0.14, corrections=2,
           land=(0.20, -0.10), hover_ms=350.0, hold_ms=160.0, reaction_ms=700.0)


def _person(params, seed, rng):
    """One try by this person: their parameters, wobbled a little."""
    return make_pointer_sample(
        seed=seed,
        start=(140.0 + rng.gauss(0, 25), 250.0 + rng.gauss(0, 15)),  # cursor starts wherever
        ms_per_bit=params["ms_per_bit"] * rng.gauss(1.0, 0.08),
        curve=max(0.0, params["curve"] * rng.gauss(1.0, 0.2)),
        overshoot=max(0.0, params["overshoot"] * rng.gauss(1.0, 0.25)),
        corrections=params["corrections"],
        land=(params["land"][0] + rng.gauss(0, 0.04), params["land"][1] + rng.gauss(0, 0.04)),
        hover_ms=max(10.0, params["hover_ms"] * rng.gauss(1.0, 0.2)),
        hold_ms=max(20.0, params["hold_ms"] * rng.gauss(1.0, 0.12)),
        reaction_ms=max(50.0, params["reaction_ms"] * rng.gauss(1.0, 0.15)),
    )


def test_two_synthetic_people_separate_under_the_scorer():
    rng = random.Random(1234)
    fit_vs = [pointer.pointer_vector(Sample.model_validate(_person(ALICE, i, rng))) for i in range(10)]
    assert all(v is not None for v in fit_vs)
    model = scorer.fit([v.values for v in fit_vs], fit_vs[0].names)

    genuine, impostor = [], []
    for i in range(10, 16):
        genuine.append(scorer.score(model, pointer.pointer_vector(
            Sample.model_validate(_person(ALICE, i, rng))).values, "pointer").score)
        impostor.append(scorer.score(model, pointer.pointer_vector(
            Sample.model_validate(_person(BOB, i, rng))).values, "pointer").score)
    assert max(genuine) < min(impostor), (sorted(genuine), sorted(impostor))
    assert min(impostor) > 2 * max(genuine)


def test_pointer_model_fits_and_flags_through_the_server(client):
    """End to end: 10 enrolments with pointer data fit a pointer model, and a login
    with someone else's pointer behaviour raises the pointer signal."""
    from helpers import PASSWORD

    rng = random.Random(5)
    client.post("/api/register", json={"username": "a", "password": PASSWORD})
    for i in range(11):  # 1 practice + 10 counted
        r = client.post("/api/enroll", json={"username": "a", "password": PASSWORD,
                                             "sample": _person(ALICE, i, rng)}).json()
    assert r["enrolled"]

    def signal(sample):
        r = client.post("/api/login", json={"username": "a", "password": PASSWORD, "sample": sample}).json()
        return next(s for s in r["signals"] if s["name"] == "pointer")

    genuine = signal(_person(ALICE, 20, rng))
    impostor = signal(_person(BOB, 21, rng))
    assert genuine["available"] and not genuine["flagged"], genuine
    assert impostor["flagged"] and impostor["score"] > 2 * genuine["score"]  # margin, not a spec: RNG-sensitive, impostor
    assert impostor["contributions"][0]["feature"].startswith("pointer.")


def test_pointer_signal_is_unavailable_when_the_form_was_submitted_with_enter(client):
    from helpers import PASSWORD

    rng = random.Random(6)
    client.post("/api/register", json={"username": "b", "password": PASSWORD})
    for i in range(11):
        client.post("/api/enroll", json={"username": "b", "password": PASSWORD,
                                         "sample": _person(ALICE, i, rng)})
    s = _person(ALICE, 30, rng)
    s["meta"]["submit_via"] = "enter"
    r = client.post("/api/login", json={"username": "b", "password": PASSWORD, "sample": s}).json()
    assert next(x for x in r["signals"] if x["name"] == "pointer")["available"] is False
