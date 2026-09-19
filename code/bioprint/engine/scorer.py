"""Anomaly scorer, generic over FeatureVectors. OWNER: Agent A (engine).

Used for keystroke vectors, pointer vectors, and by eval/cmu.py on CMU rows.

The detector is **scaled Manhattan** (Killourhy & Maxion, DSN 2009): the mean
absolute deviation from a per-feature centre, each feature divided by its own
spread. It was the strongest of the 14 detectors they compared and it still beats
most later "advanced" methods, which is exactly what we want with ten samples.

Four deliberate departures, each chosen on measured numbers, not taste. The
measurement: all 51 CMU subjects typing `.tie5Roanl`, our live layout (the Return
columns dropped, 31 features), enrolment = 10 repetitions, impostor = the first 5
attempts of each of the other 50 subjects. "EER" is the mean per-subject equal
error rate; the four columns are enrol on session 1 and log in in the same
session / the next session / sessions 5-8, and enrol practised (end of session 4)
and log in in sessions 5-8.

  | variant (centre / spread / cap)      | same session | next session | sessions 5-8 | practised |
  |--------------------------------------|--------------|--------------|--------------|-----------|
  | mean / MAD (Killourhy & Maxion)      | .125         | .166         | .239         | .084      |
  | median / MAD                         | .120         | .154         | .211         | .083      |
  | median / MAD, deviation capped at 6  | .108         | .150         | .221         | .055      |
  | ... plus the spread floors (shipped) | .109         | .151         | .221         | .055      |

  For reference, the same code on Killourhy & Maxion's own protocol (200 training
  repetitions) scores EER .075, against .096 published for scaled Manhattan and
  .103 for the stub this replaces.

  1. **Centre = median, spread = mean absolute deviation about it.** With n=10 one
     fumbled enrolment repetition moves the mean; the median it barely moves. Free
     win in every column above.
  2. **Per-feature deviation cap.** One feature that happens to be 40 spreads out
     (a phone rang mid-password) must not by itself exceed the threshold. Capping
     at `DEVIATION_CAP` spreads bounds any single feature's influence and is worth
     about 1.3 EER points in the same-session case and 3.2 in the practised case.
     It costs ~1 point in the hardest case (enrol while still learning the
     password, log in four sessions later), which we accept.
  3. **Spread floors.** A feature that was accidentally consistent across ten
     enrolments gets a near-zero spread and would then dominate every later score.
     Three floors, because this scorer also runs on pointer vectors whose features
     are in wildly different units (ratios near 1 next to milliseconds in the
     hundreds): a relative floor on |centre|, a group floor (H/DD/UD each pooled,
     so a same-unit family shares a sanity level) and an absolute epsilon.
  4. **The score is the *mean* scaled deviation, not the sum.** The unit is then
     "typical deviations", which is comparable between a 4-character and a
     16-character password and between the keystroke and pointer channels, so the
     dashboard can show score/threshold as one honest ratio.

Threshold: fitted per user from enrolment alone, by leave-one-out. Each enrolment
sample is scored against a model built from the other nine, and the threshold is a
quantile of those distances times a margin, clipped into [floor, ceiling].

The clip is not cosmetic. **At n=10 the raw leave-one-out rule lands above the
ceiling for 95% of CMU subjects** (median raw value 2.17 against a ceiling of
1.9), so in practice the ceiling sets the operating point and the per-user part
only binds for an unusually consistent enrolment - and increasingly as a profile
grows past ten samples. Sweeping LOO_QUANTILE over .75/.9/1.0 and
THRESHOLD_MARGIN over 1.5-2.25 moved no error rate by more than 0.01 for that
reason; the ceiling is the knob that matters. Measured at the shipped values:

  | scenario (enrol -> log in)                | FRR  | FAR  |
  |-------------------------------------------|------|------|
  | same session (the live demo)              | 0.04 | 0.26 |
  | next session, another day                 | 0.11 | 0.26 |
  | sessions 5-8, weeks later                 | 0.27 | 0.26 |
  | practised typist, sessions 5-8            | 0.26 | 0.03 |

Read that FAR honestly: a CMU "impostor" is a person who has been given the
password and has practised it five times, on a template built from ten
repetitions typed while the owner was still learning the password themselves.
Every constant below is a module-level knob precisely because phase 2 re-tunes
them on eval/cmu.py and on live impostor attempts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from contracts import Contribution, SignalResult

# --- detector shape ----------------------------------------------------------
USE_MEDIAN_CENTER = True   # median centre instead of the mean (robust at n=10)
DEVIATION_CAP = 6.0        # no single feature may contribute more than this many spreads
REL_SPREAD_FLOOR = 0.05    # spread >= 5% of |centre|  (unit-free; for mixed-unit vectors)
GROUP_SPREAD_FLOOR = 0.15  # spread >= 15% of the median spread of its H./DD./UD. family
ABS_SPREAD_FLOOR = 1e-6    # last-resort guard against divide-by-zero
LONE_SAMPLE_CV = 0.30      # n=1: no spread to estimate, assume a 30% coefficient of variation

# --- threshold rule ----------------------------------------------------------
LOO_QUANTILE = 0.75        # quantile of the leave-one-out enrolment distances
THRESHOLD_MARGIN = 1.5     # ... times this margin (allows for day-to-day drift)
LOO_MAX_SAMPLES = 60       # leave-one-out is O(n^2); beyond this, subsample evenly
THRESHOLD_CEILING = 1.9    # never trust a sloppy enrolment beyond this
THRESHOLD_FLOOR = 1.0      # ... nor an implausibly tidy one below it
# The ceiling is the operating point (see the header: 95% of users hit it at n=10).
# Measured on CMU, so phase 2 can move it with its eyes open — FRR is genuine users,
# FAR is impostors who were given the password and practised it five times:
#   ceiling | same session      next session      sessions 5-8      practised
#     1.5   | FRR .21 FAR .10   FRR .34 FAR .10   FRR .59 FAR .10   FRR .50 FAR .00
#     1.7   | FRR .10 FAR .18   FRR .20 FAR .18   FRR .41 FAR .18   FRR .36 FAR .01
#     1.9   | FRR .04 FAR .26   FRR .11 FAR .26   FRR .27 FAR .26   FRR .26 FAR .03
#     2.1   | FRR .03 FAR .35   FRR .06 FAR .35   FRR .18 FAR .35   FRR .19 FAR .05
# 1.9 is chosen for a demo that must not false-reject the owner on stage, and
# leans on bot.py and the pointer channel for the impostors it lets through.

# --- explanation -------------------------------------------------------------
MAX_CONTRIBUTIONS = 5      # features returned to the dashboard
MAX_REASONS = 3            # sentences shown to a human
REASON_MIN_DEVIATION = 2.0  # a feature is only worth a sentence past this many spreads
REASON_MIN_MS = 15.0       # ... and only if the difference is visible in milliseconds


@dataclass
class Model:
    names: list[str]
    center: list[float]
    spread: list[float]
    threshold: float
    n: int
    cap: float = DEVIATION_CAP        # frozen at fit time: retuning must not silently
    loo: list[float] = field(default_factory=list)  # rescore old models differently

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        # Tolerant both ways: older stored models have no `cap`/`loo` (defaults fill
        # in), and a model written by a newer build with extra keys still loads.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _family(name: str) -> str:
    """Features that share a unit and a meaning. H/DD/UD pool; anything else stands alone."""
    head = name.split(".", 1)[0]
    return head if head in ("H", "DD", "UD") else name


def _fit_arrays(X: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(X, axis=0) if USE_MEDIAN_CENTER else X.mean(axis=0)
    if len(X) < 2:
        # One sample says nothing about variability. Rather than pretend it does,
        # assume a typical coefficient of variation so the model is at least usable.
        spread = np.abs(center) * LONE_SAMPLE_CV
    else:
        spread = np.abs(X - center).mean(axis=0)
    floor = np.maximum(REL_SPREAD_FLOOR * np.abs(center), ABS_SPREAD_FLOOR)
    for fam in set(families.tolist()):
        m = families == fam
        floor[m] = np.maximum(floor[m], GROUP_SPREAD_FLOOR * np.median(spread[m]))
    return center, np.maximum(spread, floor)


def _deviations(center: np.ndarray, spread: np.ndarray, values: np.ndarray, cap: float) -> np.ndarray:
    """Per-feature scaled deviation, capped so no single feature can carry a verdict."""
    per = np.abs(values - center) / spread
    return np.minimum(per, cap) if cap else per


def fit(vectors: list[list[float]], names: list[str]) -> Model:
    """Enrol: centre, spread and a per-user threshold from these samples alone."""
    X = np.asarray(vectors, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != len(names):
        raise ValueError(f"fit expects (n_samples, {len(names)}) values, got {X.shape}")
    families = np.array([_family(n) for n in names])
    center, spread = _fit_arrays(X, families)

    # Leave-one-out: score each enrolment sample against a model built without it.
    # In-sample distances would be optimistically small (the sample helped define
    # the centre), and a threshold built on them rejects the owner's next login.
    loo: list[float] = []
    held_out = range(len(X)) if len(X) <= LOO_MAX_SAMPLES else np.linspace(
        0, len(X) - 1, LOO_MAX_SAMPLES).astype(int)  # evenly spaced: covers the whole history
    for i in held_out:
        rest = np.delete(X, i, axis=0)
        if len(rest) < 2:
            continue
        c, s = _fit_arrays(rest, families)
        loo.append(float(_deviations(c, s, X[i], DEVIATION_CAP).mean()))

    raw = float(np.quantile(loo, LOO_QUANTILE) * THRESHOLD_MARGIN) if loo else THRESHOLD_CEILING
    threshold = float(np.clip(raw, THRESHOLD_FLOOR, THRESHOLD_CEILING))
    return Model(names=list(names), center=center.tolist(), spread=spread.tolist(),
                 threshold=threshold, n=int(len(X)), cap=DEVIATION_CAP, loo=sorted(loo))


def distance(model: Model, values: list[float]) -> tuple[float, np.ndarray]:
    """Total distance and each feature's share of it.

    "Total" is the mean scaled deviation across features, so it is comparable
    across users, password lengths and signals; the shares sum to it exactly.
    """
    v = np.asarray(values, dtype=float)
    if v.shape != (len(model.names),):
        raise ValueError(f"model has {len(model.names)} features, got {v.size}")
    per = _deviations(np.asarray(model.center), np.asarray(model.spread), v, model.cap)
    return float(per.mean()), per / len(per)


# ---------------------------------------------------------------- explanation

_KEY_WORDS = {
    "Space": "space", "Period": ".", "Comma": ",", "Minus": "-", "Equal": "=",
    "Slash": "/", "Backslash": "\\", "Semicolon": ";", "Quote": "'", "Backquote": "`",
    "BracketLeft": "[", "BracketRight": "]", "CapsLock": "Caps Lock",
}


def _key_label(token: str) -> str:
    """'KeyT#0' -> 'T', 'Digit5#3' -> '5', 'Numpad5#3' -> 'numpad 5'."""
    code = token.rsplit("#", 1)[0]
    if code.startswith("Key") and len(code) == 4:
        return code[3]
    if code.startswith("Digit"):
        return code[5:]
    if code.startswith("Numpad"):
        return f"numpad {code[6:]}"
    return _KEY_WORDS.get(code, code)


def _key_phrase(token: str, qualify: set[str]) -> str:
    """How a key is named to a human. Passwords repeat letters, so when the same
    key appears twice, "held 'A' too long" is useless without saying which 'A'."""
    label = _key_label(token)
    if label in qualify and "#" in token:
        return f"'{label}' (key {int(token.rsplit('#', 1)[1]) + 1})"
    return f"'{label}'"


def _reason(name: str, value: float, expected: float, qualify: set[str] = frozenset()) -> str | None:
    """One plain-English sentence for a feature that is far from its enrolled value.

    Only positional keystroke names ("H.KeyT#0") carry enough structure - and are
    known to be in milliseconds - to phrase this way. Anything else (CMU column
    names, pointer features) gets an honest generic sentence.
    """
    parts = name.split(".")
    delta = value - expected
    if "#" in name and parts[0] in ("H", "DD", "UD"):
        if abs(delta) < REASON_MIN_MS:
            return None
        ms = abs(delta)
        if parts[0] == "H" and len(parts) == 2:
            key = _key_phrase(parts[1], qualify)
            return f"held {key} {ms:.0f} ms {'longer' if delta > 0 else 'shorter'} than usual"
        if len(parts) == 3:
            a, b = _key_phrase(parts[1], qualify), _key_phrase(parts[2], qualify)
            if parts[0] == "DD":
                when = "later" if delta > 0 else "sooner"
                return f"pressed {b} {ms:.0f} ms {when} after {a} than usual"
            if delta > 0:
                return f"paused {ms:.0f} ms longer before {b}"
            return f"ran {a} into {b} {ms:.0f} ms faster than usual"
    return f"{name} was {value:.3g}, usually about {expected:.3g}"


def score(model: Model, values: list[float], name: str = "keystroke") -> SignalResult:
    total, share = distance(model, values)
    per = share * len(share)  # back to "spreads away", which is what a human reads
    top = np.argsort(per)[::-1][:MAX_CONTRIBUTIONS]
    flagged = total > model.threshold

    # Key labels that occur more than once in this password have to be numbered.
    seen: dict[str, int] = {}
    for n in model.names:
        if n.startswith("H."):
            label = _key_label(n[2:])
            seen[label] = seen.get(label, 0) + 1
    qualify = {label for label, count in seen.items() if count > 1}

    reasons: list[str] = []
    for i in top:
        if per[i] < REASON_MIN_DEVIATION or len(reasons) >= MAX_REASONS:
            continue
        if r := _reason(model.names[i], values[i], model.center[i], qualify):
            reasons.append(r)
    if not reasons:
        reasons = [
            "many small timing differences across the whole password, none decisive on its own"
            if flagged else
            f"timing matched the enrolled rhythm ({total / model.threshold:.0%} of the limit)"
        ]

    return SignalResult(
        name=name,
        score=total,
        threshold=model.threshold,
        flagged=flagged,
        contributions=[
            Contribution(feature=model.names[i], value=values[i], expected=model.center[i],
                         deviation=float(share[i]))
            for i in top
        ],
        reasons=reasons,
    )
