"""Scrambled-keypad captcha features. OWNER: Agent L (keypad).

A keypad whose digits are shuffled on every challenge. Because the arrangement is
new each time, tapping a digit is not a memorised motor program: it is a **visual
search** ("where is the 3?") followed by a **reach** and a **press**. Every tap is
therefore a behavioural sample, and — unlike keystroke timing — one that a thumb
on a phone produces just as readily as a mouse on a laptop.

The sample unit is the TAP, not the captcha. Six captchas of six digits is ~36
samples, which is what makes a model fittable inside a signup flow.

Two halves, scored **separately** and never merged:

  cognitive ("keypad", NAMES_COG)
      How long the search-and-reach takes and how the press is timed. Built only
      from quantities a thumb can also produce, so it is scored across device
      classes: enrol on the laptop, verify on the phone.

  motor ("keypad_motor", NAMES_MOTOR)
      How the hand actually moved. Only meaningful when the device class matches
      enrolment (a mouse trajectory and a thumb tap are not comparable), so this
      half is **advisory** and is simply unavailable on a class change.

This split is the same rule as the project's "templates are per (user, device
class)" convention, applied one level down: the part of the behaviour that
survives a device change is kept, the part that does not is fenced off rather
than quietly poisoning the score.

Cognitive features (per tap, prefix "kp."):
  interval_ms       previous tap's release -> this press (shown_at -> press for
                    the first tap). The whole search + reach + decide cycle.
                    Cross-device: a thumb and a mouse both take time to find a
                    digit that moved. Capped at CAP_MS.
  interval_per_bit  interval_ms / log2(distance / key_width + 1), where distance
                    is from the previous tap point (from the centre of the
                    "target" box for the first tap). Fitts's index of difficulty
                    removes the part of the interval that is just geometry — how
                    far this scramble happened to put the next key — leaving the
                    person's own search-and-reach rate. Without it, an unlucky
                    layout looks like an impostor.
  hold_ms           press -> release. A press duration exists on every input
                    device (touchscreens report it honestly, unlike the 0 ms
                    holds a virtual *keyboard* reports). Capped at CAP_MS.
  hold_frac         hold_ms / (hold_ms + interval_ms). Unit-free: what share of
                    the tap cycle the finger spends down. It separates "slow and
                    deliberate throughout" from "long hunt, stabbed quickly",
                    which the two raw times alone do not, and it is invariant to
                    someone simply being in a hurry today.

Motor features (per tap, prefix "kpm.", device-class specific):
  mouse   the reach kinematics of engine/pointer.py, applied to the key's rect
          instead of the submit button: ms_per_bit, path_efficiency, curvature,
          peak_mean_ratio, speed_variation, overshoot, land_dx, land_dy,
          hover_ms, settle_ms, hold_ms.
  touch   only land_dx, land_dy and hold_ms. A finger has no pre-tap path — the
          first thing the screen sees is the contact — so there is nothing else
          honestly measurable. Where on the key a person's thumb lands is a real
          and well-known individual bias; that is all we claim.

Taps that are not digits (backspace, a blank cell) are not samples, but they
still count as "the previous tap" for the next interval: the time spent hitting
backspace is real time the next search did not take.

A tap whose reach is too short to measure gets no motor vector: too few captured
moves (pointer.MIN_MOVES) or a start already on the key, which is what a repeated
digit looks like. The cognitive vector is still produced for those taps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts import (
    KEYPAD_BACK,
    Contribution,
    DeviceClass,
    FeatureVector,
    KeypadRun,
    KeypadTap,
    PointerEvent,
    SignalResult,
)
from engine import bot, pointer, scorer

# --- feature names -----------------------------------------------------------

NAMES_COG: list[str] = [
    "kp.interval_ms",
    "kp.interval_per_bit",
    "kp.hold_ms",
    "kp.hold_frac",
]

NAMES_MOTOR: dict[DeviceClass, list[str]] = {
    "mouse": [
        "kpm.ms_per_bit",
        "kpm.path_efficiency",
        "kpm.curvature",
        "kpm.peak_mean_ratio",
        "kpm.speed_variation",
        "kpm.overshoot",
        "kpm.land_dx",
        "kpm.land_dy",
        "kpm.hover_ms",
        "kpm.settle_ms",
        "kpm.hold_ms",
    ],
    "touch": [
        "kpm.land_dx",
        "kpm.land_dy",
        "kpm.hold_ms",
    ],
    "unknown": [],
}

# --- knobs -------------------------------------------------------------------
CAP_MS = 5000.0            # cap on every duration feature (one distracted tap must not dominate)
MIN_BITS = 1.0             # floor on the Fitts index of difficulty; see _cog_values
DEFAULT_KEY_PX = 60.0      # fallback key size when a rect is missing from run.cells
HOLD_FALLBACK_MS = 80.0    # a tap with no recorded release: use the run's median hold, else this
MIN_MOTOR_RUNS = 3         # fewer matching-class runs than this: no motor model at all
MIN_MOTOR_VECTORS = 8      # ... nor with fewer reach samples than this

# --- bot rules (see bot_check) ----------------------------------------------
FIRST_TAP_MS = 200.0       # first tap that soon after the layout appeared: suspicious
IDENTICAL_TOL_MS = 0.5     # "every interval the same": within timer resolution
IDENTICAL_MIN_N = 3
SHORT_HOLD_MIN_N = 2       # holds under bot.HOLD_FLOOR_MS needed to call it mechanical
PIXEL_PERFECT_PX = 0.5     # landing this close to the exact key centre is a script
PIXEL_PERFECT_MIN_N = 3


# ---------------------------------------------------------------- small helpers


def _finite(x: float, fallback: float = 0.0) -> float:
    x = float(x)
    return x if math.isfinite(x) else fallback


def _cap(ms: float) -> float:
    return float(min(max(ms, 0.0), CAP_MS))


def _cell_key(cell: int) -> str:
    """run.cells is keyed by cell index as a string, plus "back" and "target"."""
    return "back" if cell < 0 else str(cell)


def _rect(run: KeypadRun, cell: int) -> list[float] | None:
    r = run.cells.get(_cell_key(cell))
    if not r or len(r) != 4 or r[2] <= 0 or r[3] <= 0:
        return None
    return [float(v) for v in r]


def _centre(rect: list[float]) -> tuple[float, float]:
    return rect[0] + rect[2] / 2, rect[1] + rect[3] / 2


def _is_digit(tap: KeypadTap) -> bool:
    return 0 <= tap.digit <= 9


def _release(tap: KeypadTap) -> float:
    return float(tap.t_up if tap.t_up is not None else tap.t_down)


def _median_hold(run: KeypadRun) -> float:
    holds = [float(t.t_up) - t.t_down for t in run.taps if t.t_up is not None and t.t_up >= t.t_down]
    return float(np.median(holds)) if holds else HOLD_FALLBACK_MS


def _hold_ms(tap: KeypadTap, fallback: float) -> float:
    if tap.t_up is None:
        return _cap(fallback)
    return _cap(float(tap.t_up) - tap.t_down)


def device_class(run: KeypadRun) -> DeviceClass:
    """The run's device class, from the majority pointer_type of its taps.

    Anything that is neither mouse nor touch (a pen, a browser that reports "")
    is "unknown": the cognitive half still scores, the motor half does not.
    """
    counts: dict[str, int] = {}
    for tap in run.taps:
        counts[tap.pointer_type] = counts.get(tap.pointer_type, 0) + 1
    if not counts:
        return "unknown"
    top = max(counts.items(), key=lambda kv: (kv[1], kv[0] == "mouse"))[0]
    return top if top in ("mouse", "touch") else "unknown"


# ---------------------------------------------------------------- feature extraction


def _cog_values(run: KeypadRun, tap: KeypadTap, prev: KeypadTap | None, hold_fallback: float) -> list[float]:
    """One cognitive vector. `prev` is the tap before this one, digit or not."""
    if prev is None:
        origin = run.cells.get("target")
        start = _centre([float(v) for v in origin]) if origin and len(origin) == 4 else (tap.x, tap.y)
        t0 = float(run.shown_at)
    else:
        start = (prev.x, prev.y)
        t0 = _release(prev)

    interval = _cap(tap.t_down - t0)
    hold = _hold_ms(tap, hold_fallback)

    rect = _rect(run, tap.cell)
    width = float(rect[2]) if rect else DEFAULT_KEY_PX
    dist = math.hypot(tap.x - start[0], tap.y - start[1])
    # Floored: a target within its own width is not an easier search, it is *no*
    # search (the same digit twice). Without the floor those taps divide by ~0 and
    # the feature's spread blows up, drowning every real hop.
    bits = max(math.log2(dist / width + 1.0) if width > 0 else 0.0, MIN_BITS)
    per_bit = interval / bits

    total = hold + interval
    frac = hold / total if total > 0 else 0.0
    return [_finite(interval), _finite(min(per_bit, CAP_MS)), _finite(hold), _finite(frac)]


def _approach(moves: list[PointerEvent], t_from: float, t_to: float) -> list[PointerEvent]:
    """Captured moves belonging to this tap's reach.

    Sliced by time rather than by pointer.EPISODE_GAP_MS: on a keypad the taps are
    a few hundred ms apart, so a rest-gap heuristic would happily merge two reaches.
    """
    return [m for m in moves if t_from <= m.t <= t_to]


def _motor_mouse(tap: KeypadTap, rect: list[float], moves: list[PointerEvent],
                 hold_fallback: float) -> list[float] | None:
    """Reach kinematics for one tap. The shape of engine/pointer.pointer_vector,
    with the key's rect as the target. None when there is no measurable reach."""
    if len(moves) < pointer.MIN_MOVES:
        return None
    rx, ry, rw, rh = rect
    cx, cy = _centre(rect)

    t = np.array([m.t for m in moves] + [tap.t_down], dtype=float)
    xy = np.array([[m.x, m.y] for m in moves] + [[tap.x, tap.y]], dtype=float)
    t = t + np.arange(len(t)) * 1e-6  # coalesced events can tie; interp needs strictly increasing

    start = xy[0]
    dist = float(math.hypot(cx - start[0], cy - start[1]))
    if dist < 0.5 * min(rw, rh):
        return None  # already on the key: a repeated digit, no reach to measure

    t_end = float(t[-2])  # last recorded move: the movement proper ends here
    approach = t_end - float(t[0])
    if approach <= 0:
        return None

    inside = np.array([pointer._inside(px, py, rect) for px, py in xy])
    i_arr = len(xy) - 1
    while i_arr > 0 and inside[i_arr - 1]:
        i_arr -= 1
    t_arr = float(t[i_arr])

    ux, uy = (cx - start[0]) / dist, (cy - start[1]) / dist
    width = min(rw / abs(ux) if abs(ux) > 1e-9 else math.inf,
                rh / abs(uy) if abs(uy) > 1e-9 else math.inf)
    bits = math.log2(dist / width + 1.0)

    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    path = float(seg.sum())
    straight = float(np.hypot(*(xy[-1] - start)))
    efficiency = straight / path if path > 0 else 1.0
    rel = xy - start
    perp = np.abs(rel[:, 0] * uy - rel[:, 1] * ux)
    curvature = float(perp.max()) / dist
    along = rel[:, 0] * ux + rel[:, 1] * uy
    overshoot = (float(along.max()) - dist) / dist

    v = pointer._speed_profile(t[:-1], xy[:-1])
    mean_v = float(v.mean())
    i_peak = int(np.argmax(v))
    peak_mean = float(v[i_peak]) / mean_v if mean_v > 0 else 1.0
    variation = float(np.abs(np.diff(v[i_peak:])).sum()) / float(v[i_peak]) if v[i_peak] > 0 else 0.0

    values = [
        min(approach / bits, CAP_MS) if bits > 1e-6 else CAP_MS,
        efficiency,
        curvature,
        peak_mean,
        variation,
        overshoot,
        (tap.x - cx) / rw,
        (tap.y - cy) / rh,
        _cap(tap.t_down - t_arr),
        _cap(tap.t_down - t_end),
        _hold_ms(tap, hold_fallback),
    ]
    return [_finite(x) for x in values]


def _motor_touch(tap: KeypadTap, rect: list[float], hold_fallback: float) -> list[float] | None:
    cx, cy = _centre(rect)
    return [
        _finite((tap.x - cx) / rect[2]),
        _finite((tap.y - cy) / rect[3]),
        _finite(_hold_ms(tap, hold_fallback)),
    ]


def tap_vectors(run: KeypadRun) -> tuple[list[FeatureVector], list[FeatureVector]]:
    """(cognitive, motor) vectors, one per digit tap.

    The two lists are NOT parallel: a tap with no measurable reach contributes a
    cognitive vector and no motor one. They feed two independent models.
    """
    cls = device_class(run)
    motor_names = NAMES_MOTOR.get(cls, [])
    hold_fallback = _median_hold(run)
    moves = sorted((p for p in run.pointer if p.type == "move"), key=lambda p: p.t)

    cog: list[FeatureVector] = []
    motor: list[FeatureVector] = []
    prev: KeypadTap | None = None
    for tap in run.taps:
        if not _is_digit(tap):
            prev = tap  # backspace / blank: not a sample, but the clock kept running
            continue
        cog.append(FeatureVector(names=list(NAMES_COG), values=_cog_values(run, tap, prev, hold_fallback)))

        rect = _rect(run, tap.cell)
        if rect and motor_names:
            t_from = float(run.shown_at) if prev is None else _release(prev)
            if cls == "mouse":
                vals = _motor_mouse(tap, rect, _approach(moves, t_from, tap.t_down), hold_fallback)
            else:
                vals = _motor_touch(tap, rect, hold_fallback)
            if vals is not None:
                motor.append(FeatureVector(names=list(motor_names), values=vals))
        prev = tap
    return cog, motor


def entered_digits(run: KeypadRun) -> list[int]:
    """What the taps actually put in the entry box, backspaces applied."""
    entry: list[int] = []
    for tap in run.taps:
        if tap.digit == KEYPAD_BACK:
            if entry:
                entry.pop()
        elif _is_digit(tap) and len(entry) < len(run.target):
            entry.append(tap.digit)
    return entry


def run_stats(run: KeypadRun) -> dict:
    """Per-run counters for the dashboard. Not features: these are for a human."""
    digits = [t for t in run.taps if _is_digit(t)]
    last = max((_release(t) for t in run.taps), default=float(run.shown_at))
    return {
        "taps": len(digits),
        "errors": sum(1 for t in digits if not t.correct),
        "backspaces": sum(1 for t in run.taps if t.digit == KEYPAD_BACK),
        "duration_ms": _finite(last - run.shown_at),
        "first_tap_ms": _finite(run.taps[0].t_down - run.shown_at) if run.taps else 0.0,
        "completed": bool(run.completed),
        "device_class": device_class(run),
    }


# ---------------------------------------------------------------- profile


@dataclass
class Profile:
    cog: scorer.Model
    motor: scorer.Model | None
    device_class: DeviceClass
    n_runs: int

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "cog": self.cog.to_dict(),
            "motor": self.motor.to_dict() if self.motor is not None else None,
            "device_class": self.device_class,
            "n_runs": self.n_runs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        motor = d.get("motor")
        return cls(
            cog=scorer.Model.from_dict(d["cog"]),
            motor=scorer.Model.from_dict(motor) if motor else None,
            device_class=d.get("device_class", "unknown"),
            n_runs=int(d.get("n_runs", 0)),
        )


def _run_distance(model: scorer.Model, vectors: list[FeatureVector]) -> float | None:
    """A captcha's score: the mean per-tap distance. See _run_threshold."""
    ds = [scorer.distance(model, v.values)[0] for v in vectors]
    return float(np.mean(ds)) if ds else None


def _run_threshold(per_run: list[list[FeatureVector]], names: list[str]) -> tuple[float, list[float]]:
    """Leave-one-*run*-out threshold.

    scorer.fit's own threshold is fitted on single samples, and here a sample is
    one tap. But nothing is ever judged on one tap: a captcha is scored as the
    mean over ~6-12 taps, and a mean of 8 draws is roughly sqrt(8) times tighter
    than one draw. A tap-level threshold would therefore be far too lenient —
    an impostor whose every tap sits 1.5 spreads out would pass a threshold
    fitted to admit single taps 1.9 spreads out. So hold out a whole run, fit on
    the rest, and score the held-out run the way a live run will be scored.
    """
    loo: list[float] = []
    if len(per_run) >= 2:
        for i, held in enumerate(per_run):
            rest = [v.values for j, r in enumerate(per_run) if j != i for v in r]
            if len(rest) < 2 or not held:
                continue
            m = scorer.fit(rest, names)
            if (d := _run_distance(m, held)) is not None:
                loo.append(d)
    raw = (float(np.quantile(loo, scorer.LOO_QUANTILE) * scorer.THRESHOLD_MARGIN)
           if loo else scorer.THRESHOLD_CEILING)
    threshold = float(np.clip(raw, scorer.THRESHOLD_FLOOR, scorer.THRESHOLD_CEILING))
    return threshold, sorted(loo)


def fit_profile(runs: list[KeypadRun]) -> Profile:
    """Enrol: one cognitive model over every tap of every run, plus a motor model
    for the majority device class if there is enough of it."""
    if not runs:
        raise ValueError("fit_profile needs at least one run")

    classes = [device_class(r) for r in runs]
    ranked = sorted(set(classes), key=lambda c: (-classes.count(c), c == "unknown"))
    majority: DeviceClass = ranked[0]

    per_run_cog: list[list[FeatureVector]] = []
    per_run_motor: list[list[FeatureVector]] = []
    for run, cls in zip(runs, classes):
        cog, motor = tap_vectors(run)
        per_run_cog.append(cog)
        per_run_motor.append(motor if cls == majority else [])

    flat = [v.values for r in per_run_cog for v in r]
    if not flat:
        raise ValueError("fit_profile found no usable taps")
    cog_model = scorer.fit(flat, NAMES_COG)
    cog_model.threshold, cog_model.loo = _run_threshold(per_run_cog, NAMES_COG)

    motor_model: scorer.Model | None = None
    motor_names = NAMES_MOTOR.get(majority, [])
    with_motor = [m for m in per_run_motor if m]
    flat_motor = [v.values for m in per_run_motor for v in m]
    if motor_names and len(with_motor) >= MIN_MOTOR_RUNS and len(flat_motor) >= MIN_MOTOR_VECTORS:
        motor_model = scorer.fit(flat_motor, motor_names)
        motor_model.threshold, motor_model.loo = _run_threshold(per_run_motor, motor_names)

    return Profile(cog=cog_model, motor=motor_model, device_class=majority, n_runs=len(runs))


# ---------------------------------------------------------------- scoring


def _pct(value: float, expected: float) -> str:
    return f"{abs(value - expected) / abs(expected) * 100:.0f}%" if abs(expected) > 1e-9 else "much"


def _reason(name: str, value: float, expected: float) -> str | None:
    """One sentence a judge can read. Never jargon, never a feature name if avoidable."""
    d = value - expected
    ms = abs(d)
    if name in ("kp.interval_ms", "kp.hold_ms", "kpm.hover_ms", "kpm.settle_ms", "kpm.hold_ms") \
            and ms < scorer.REASON_MIN_MS:
        return None
    match name:
        case "kp.interval_ms":
            return (f"took {ms:.0f} ms longer than usual to find and reach each digit" if d > 0 else
                    f"found and reached each digit {ms:.0f} ms faster than usual")
        case "kp.interval_per_bit":
            return (f"searched and reached {_pct(value, expected)} "
                    f"{'slower' if d > 0 else 'faster'} than usual, allowing for how far apart "
                    "this layout put the keys")
        case "kp.hold_ms":
            return f"pressed keys {ms:.0f} ms {'longer' if d > 0 else 'shorter'} than usual"
        case "kp.hold_frac":
            return (f"spent {value:.0%} of each tap with the key held down, against "
                    f"{expected:.0%} at signup")
        case "kpm.ms_per_bit":
            return f"moved to each key {_pct(value, expected)} {'slower' if d > 0 else 'faster'} than usual"
        case "kpm.path_efficiency":
            return ("took a straighter route to each key than usual" if d > 0 else
                    "wandered more on the way to each key than usual")
        case "kpm.curvature":
            return f"the arc of each reach was {'wider' if d > 0 else 'flatter'} than usual"
        case "kpm.peak_mean_ratio" | "kpm.speed_variation":
            return ("accelerated and corrected differently on the way to each key "
                    f"({value:.2f} against {expected:.2f} at signup)")
        case "kpm.overshoot":
            return ("went past each key before pressing more than usual" if d > 0 else
                    "stopped shorter of each key than usual")
        case "kpm.land_dx" | "kpm.land_dy":
            axis = "right of" if name.endswith("dx") and d > 0 else \
                   "left of" if name.endswith("dx") else \
                   "below" if d > 0 else "above"
            return f"pressed {abs(d):.0%} of a key {axis} the spot usually pressed"
        case "kpm.hover_ms":
            return f"hovered on each key {ms:.0f} ms {'longer' if d > 0 else 'less'} before pressing"
        case "kpm.settle_ms":
            return f"the pointer sat still {ms:.0f} ms {'longer' if d > 0 else 'less'} before each press"
        case "kpm.hold_ms":
            return f"held each press {ms:.0f} ms {'longer' if d > 0 else 'shorter'} than usual"
    return f"{name} was {value:.3g}, usually about {expected:.3g}"


def _score(model: scorer.Model, vectors: list[FeatureVector], name: str, advisory: bool) -> SignalResult:
    X = np.array([v.values for v in vectors], dtype=float)
    shares = np.array([scorer.distance(model, v.values)[1] for v in vectors], dtype=float)
    share = shares.mean(axis=0)
    total = float(share.sum())
    per = share * len(share)  # mean "spreads away", which is what a human reads
    observed = X.mean(axis=0)
    flagged = total > model.threshold

    top = np.argsort(per)[::-1][:scorer.MAX_CONTRIBUTIONS]
    reasons: list[str] = []
    for i in top:
        if per[i] < scorer.REASON_MIN_DEVIATION or len(reasons) >= scorer.MAX_REASONS:
            continue
        if r := _reason(model.names[i], float(observed[i]), float(model.center[i])):
            reasons.append(f"{r} (advisory)" if advisory else r)
    if not reasons:
        reasons = [("several small differences across the whole captcha, none decisive on its own"
                    if flagged else
                    f"keypad behaviour matched the enrolled profile "
                    f"({total / model.threshold:.0%} of the limit)")]
        if advisory:
            reasons = [f"{reasons[0]} (advisory)"]

    return SignalResult(
        name=name,
        available=True,
        score=total,
        threshold=model.threshold,
        flagged=flagged,
        contributions=[
            Contribution(feature=model.names[i], value=float(observed[i]),
                         expected=float(model.center[i]), deviation=float(share[i]))
            for i in top
        ],
        reasons=reasons,
    )


def _unavailable(name: str, reason: str) -> SignalResult:
    return SignalResult(name=name, available=False, score=0.0, threshold=1.0, flagged=False,
                        reasons=[reason])


def score_runs(profile: Profile, runs: list[KeypadRun]) -> tuple[SignalResult, SignalResult]:
    """Score the captchas just solved against the enrolled profile.

    Returns ("keypad", "keypad_motor"). The two are never combined here: the
    cognitive half is the verdict, the motor half is corroboration that is simply
    absent on a different kind of device.
    """
    cog_vs: list[FeatureVector] = []
    motor_vs: list[FeatureVector] = []
    classes: list[DeviceClass] = []
    for run in runs:
        c, m = tap_vectors(run)
        cog_vs += c
        motor_vs += m
        classes.append(device_class(run))
    here: DeviceClass = classes[0] if len(set(classes)) == 1 else ("unknown" if not classes else classes[-1])

    if not cog_vs or len(profile.cog.names) != len(NAMES_COG):
        return (_unavailable("keypad", "no usable keypad taps in this attempt"),
                _unavailable("keypad_motor", "no usable keypad taps in this attempt"))
    cog = _score(profile.cog, cog_vs, "keypad", advisory=False)

    if profile.motor is None:
        motor = _unavailable("keypad_motor", "no movement profile was recorded at signup")
    elif here != profile.device_class:
        motor = _unavailable(
            "keypad_motor",
            f"enrolled on {_class_word(profile.device_class)}, this is "
            f"{_class_word(here)}: movement not comparable")
    elif not motor_vs or len(motor_vs[0].values) != len(profile.motor.names):
        motor = _unavailable("keypad_motor", "not enough movement was captured to judge")
    else:
        motor = _score(profile.motor, motor_vs, "keypad_motor", advisory=True)
    return cog, motor


def _class_word(cls: DeviceClass) -> str:
    return {"mouse": "a mouse", "touch": "a touch screen", "unknown": "an unrecognised device"}[cls]


# ---------------------------------------------------------------- bot rules


def _kp_trust(runs: list[KeypadRun], acc: bot._Acc) -> None:
    n = sum(not t.trusted for r in runs for t in r.taps)
    if n:
        acc.add("keypad_untrusted_taps", bot.STRONG,
                f"{n} keypad taps were generated by a script (isTrusted=false)", n)
    n = sum(not e.trusted for r in runs for e in r.pointer)
    if n:
        acc.add("keypad_untrusted_pointer", bot.STRONG,
                f"{n} pointer events on the keypad were generated by a script (isTrusted=false)", n)


def _kp_intervals(runs: list[KeypadRun], acc: bot._Acc) -> None:
    """Check mechanical repetition, not presumed visual-search speed.

    The whole target and layout remain visible for a run. A person can plan
    later taps during earlier holds, so release-to-next-press gaps are not
    independent reaction times and cannot establish automation by being short.
    """
    all_intervals: list[float] = []
    for run in runs:
        prev: KeypadTap | None = None
        for tap in run.taps:
            if prev is not None:
                gap = tap.t_down - _release(prev)
                all_intervals.append(gap)
            prev = tap
    if len(all_intervals) >= IDENTICAL_MIN_N and \
            max(all_intervals) - min(all_intervals) <= IDENTICAL_TOL_MS:
        acc.add("keypad_identical_intervals", bot.STRONG,
                f"every gap between taps is identical ({all_intervals[0]:.1f} ms): mechanical input", 0.0)


def _kp_first_tap(runs: list[KeypadRun], acc: bot._Acc) -> None:
    quick = [r.taps[0].t_down - r.shown_at for r in runs
             if r.taps and r.taps[0].t_down - r.shown_at < FIRST_TAP_MS]
    if quick:
        acc.add("keypad_instant_start", 0.4,
                f"first tap came {min(quick):.0f} ms after the keypad appeared, before a person "
                "could have read the digits to enter", min(quick), FIRST_TAP_MS)


def _kp_holds(runs: list[KeypadRun], acc: bot._Acc) -> None:
    short = [float(t.t_up) - t.t_down for r in runs for t in r.taps
             if t.t_up is not None and float(t.t_up) - t.t_down < bot.HOLD_FLOOR_MS]
    if len(short) >= SHORT_HOLD_MIN_N:
        acc.add("keypad_impossible_hold", bot.STRONG,
                f"{len(short)} taps held under {bot.HOLD_FLOOR_MS:.0f} ms "
                f"(min {min(short):.1f} ms): shorter than a real press", min(short), bot.HOLD_FLOOR_MS)


def _kp_landing(runs: list[KeypadRun], acc: bot._Acc) -> None:
    exact = 0
    for run in runs:
        for tap in run.taps:
            rect = _rect(run, tap.cell)
            if rect is None:
                continue
            cx, cy = _centre(rect)
            if math.hypot(tap.x - cx, tap.y - cy) <= PIXEL_PERFECT_PX:
                exact += 1
    if exact >= PIXEL_PERFECT_MIN_N:
        acc.add("keypad_pixel_perfect", bot.STRONG,
                f"{exact} taps landed exactly on the centre of the key: a hand never repeats "
                "a pixel", exact, PIXEL_PERFECT_MIN_N)


def _kp_movement(runs: list[KeypadRun], acc: bot._Acc) -> None:
    for run in runs:
        if device_class(run) == "mouse" and run.taps and \
                not any(p.type == "move" for p in run.pointer):
            acc.add("keypad_no_movement", bot.STRONG,
                    "a mouse tapped every key without the pointer moving once", 0.0)
            return


def _kp_device(runs: list[KeypadRun], acc: bot._Acc) -> None:
    for run in runs:
        cls = device_class(run)
        env = run.env if isinstance(run.env, dict) else {}
        if cls == "touch" and env.get("max_touch_points") == 0:
            acc.add("keypad_touch_without_touchscreen", 0.4,
                    "taps claim to come from a touch screen on a device that has none")
            return
        if cls == "mouse" and env.get("ua_mobile") is True:
            acc.add("keypad_mouse_on_phone", 0.4,
                    "taps claim to come from a mouse on a device that says it is a phone")
            return


def _kp_entry(runs: list[KeypadRun], acc: bot._Acc) -> None:
    bad = sum(1 for r in runs if not r.completed or entered_digits(r) != list(r.target))
    if bad:
        acc.add("keypad_not_solved", 0.4,
                f"{bad} captcha(s) were submitted without the digits actually being entered",
                bad, 0.0)


def bot_check(runs: list[KeypadRun]) -> SignalResult:
    """Non-human detection for the keypad, with bot.py's accumulator semantics:
    a STRONG rule blocks on its own, weak tells must stack past bot.THRESHOLD.

    The keypad is a good place to catch a script precisely because the layout is
    new every time: an automated solver must read the DOM, and reading is instant.
    """
    acc = bot._Acc()
    if not runs:
        acc.add("keypad_no_runs", 0.4, "no keypad captchas were submitted")
        return acc.result()
    for rule in (lambda: _kp_trust(runs, acc),
                 lambda: bot._env(runs[-1].env if isinstance(runs[-1].env, dict) else {}, acc),
                 lambda: _kp_intervals(runs, acc),
                 lambda: _kp_first_tap(runs, acc),
                 lambda: _kp_holds(runs, acc),
                 lambda: _kp_landing(runs, acc),
                 lambda: _kp_movement(runs, acc),
                 lambda: _kp_device(runs, acc),
                 lambda: _kp_entry(runs, acc)):
        try:
            rule()
        except Exception as exc:  # a malformed run must not crash a login; say so instead
            acc.add("probe_error", 0.0, f"keypad bot rule skipped: {type(exc).__name__}")
    return acc.result()
