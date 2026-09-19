"""Shared data contracts. Every module and every agent codes against these.

Change a shape here only by agreement: the browser, the engine, the CMU
evaluation and the dashboard all depend on it.

Time: every `t` in a sample is milliseconds since the same origin, the moment the
form was shown or reset (performance.now() at that instant). Keystrokes and
pointer events therefore share one clock, so "last key released -> click" is a
plain subtraction.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_EVENTS = 20_000  # a login is ~40 key events and a few hundred pointer moves

# ---------------------------------------------------------------- browser -> server


class KeyEvent(BaseModel):
    code: str = Field(max_length=32)  # event.code: physical key, e.g. "KeyT", "ShiftLeft"
    type: Literal["down", "up"]
    t: float
    field: Literal["username", "password", "other"] = "other"
    trusted: bool = True  # event.isTrusted; false means script-dispatched


class PointerEvent(BaseModel):
    type: Literal["move", "down", "up", "enter", "leave"]
    t: float
    x: float  # clientX
    y: float  # clientY
    pointer_type: str = Field(default="mouse", max_length=16)  # mouse | pen | touch
    buttons: int = 0
    target: str | None = Field(default=None, max_length=32)  # element id under the pointer, if one we care about
    trusted: bool = True


class Meta(BaseModel):
    had_paste: bool = False
    viewport: str = Field(default="", max_length=32)  # "1280x720@2"
    # Bounding boxes in client coordinates at submit time, [x, y, w, h].
    # "submit" is always the button that submitted THIS sample (Log in or Enroll),
    # so pointer features compare like with like. Also "username", "password".
    targets: dict[str, list[float]] = Field(default_factory=dict)
    submit_via: Literal["click", "enter", "unknown"] = "unknown"


class Sample(BaseModel):
    """One typing of the credentials: an enrollment repetition or a login attempt."""

    keystrokes: list[KeyEvent] = Field(max_length=MAX_EVENTS)
    pointer: list[PointerEvent] = Field(default_factory=list, max_length=MAX_EVENTS)
    env: dict[str, Any] = Field(default_factory=dict)  # static/probe.js output; bot.py owns the keys
    meta: Meta = Field(default_factory=Meta)


class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class AttemptIn(BaseModel):
    """Body of both /api/enroll and /api/login."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    sample: Sample


STEP_UP_SAMPLES = 3  # extra typings asked for after a "step_up" decision
STEP_UP_WINDOW_S = 10 * 60  # a pending step_up attempt older than this is forgotten


class StepUpIn(BaseModel):
    """Body of /api/login/stepup: the answer to a "step_up" decision.

    There is no token. The server re-checks the password and takes the user's most
    recent login attempt with decision "step_up" (within STEP_UP_WINDOW_S, and not
    followed by an allow/block) as the pending first sample. Its stored keystroke
    score joins the three new ones; the median of the four is the verdict.
    This is more of the same behavioural evidence, not a second factor.
    """

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    samples: list[Sample] = Field(min_length=STEP_UP_SAMPLES, max_length=STEP_UP_SAMPLES)


# ---------------------------------------------------------------- engine outputs


class FeatureVector(BaseModel):
    """Named numbers. `names` must be identical across every sample scored by one
    model: same password, same keying, same order. CMU rows convert to this too."""

    names: list[str]
    values: list[float]


class Contribution(BaseModel):
    """How much one feature pushed a score. The dashboard's explanation is built from these."""

    feature: str  # e.g. "H.KeyT#2", "DD.KeyT#2.KeyI#3", "pointer.path_efficiency"
    value: float  # observed
    expected: float  # enrolled centre
    deviation: float  # this feature's share of the score, same units as SignalResult.score


class SignalResult(BaseModel):
    """Uniform result for every signal. Signals are never merged into one number here."""

    name: Literal["keystroke", "pointer", "bot", "device"]
    available: bool = True  # False: not enough data to judge (e.g. no pointer used)
    score: float = 0.0  # larger = less like the owner (or more bot-like)
    threshold: float = 1.0
    flagged: bool = False  # score > threshold, or a hard rule fired
    contributions: list[Contribution] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # human-readable, shown as-is


# "step_up": password right, rhythm right, but the device looks new. Not a login
# yet: the browser is asked for STEP_UP_SAMPLES more typings (POST /api/login/stepup)
# and the median keystroke score of all four decides allow/block.
Decision = Literal["allow", "block", "step_up", "retype", "wrong_password", "unknown_user", "not_enrolled"]


class LoginOut(BaseModel):
    decision: Decision
    reasons: list[str]
    signals: list[SignalResult] = Field(default_factory=list)
    latency_ms: float
    attempt_id: int | None = None


class EnrollOut(BaseModel):
    accepted: bool  # False when the sample was rejected (wrong password, retype needed)
    count: int  # accepted enrollment samples so far
    target: int
    enrolled: bool
    warmup: bool = False  # this sample was a practice run and did not count
    reasons: list[str] = Field(default_factory=list)
