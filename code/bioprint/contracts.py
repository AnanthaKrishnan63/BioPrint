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


# ---------------------------------------------------------------- scrambled keypad
# A numeric captcha on a keypad whose digits are shuffled every time. Each tap
# needs a visual search, a reach and a press, so one captcha yields
# KEYPAD_TARGET_LEN behavioural samples on ANY device, touch included. Enrolled
# on the laptop after the password reps; used as the step-up on a new device.
#
# Time: every t in a run (shown_at, taps, pointer) is ms since one origin taken
# when the run's page was shown, exactly like Sample.

KEYPAD_DIGITS = 5  # keys show 0..KEYPAD_DIGITS-1; set to 10 for a full keypad
KEYPAD_COLS = 3
KEYPAD_ROWS = 2  # KEYPAD_COLS * KEYPAD_ROWS cells; cells beyond KEYPAD_DIGITS are blank (-1)
KEYPAD_TARGET_LEN = 6  # digits per captcha
KEYPAD_ENROLL_RUNS = 6  # captchas solved at signup
KEYPAD_STEPUP_RUNS = 2  # captchas solved on a "keypad" decision
KEYPAD_CHALLENGE_TTL_S = 5 * 60
KEYPAD_BACK = -1  # `digit` of the backspace key
KEYPAD_BLANK = -2  # `digit` of an empty cell

DeviceClass = Literal["mouse", "touch", "unknown"]


class KeypadChallenge(BaseModel):
    """Server -> browser. Issued by POST /api/keypad/challenge; the run must echo `id`."""

    id: str
    layout: list[int]  # cell index (row-major) -> digit, or KEYPAD_BLANK
    target: list[int]  # digits to enter, in order
    cols: int = KEYPAD_COLS
    rows: int = KEYPAD_ROWS
    expires_at: str


class KeypadTap(BaseModel):
    """One press on the keypad, in the order it happened. Wrong taps are kept."""

    t_down: float
    t_up: float | None = None
    cell: int  # index into layout, or -1 for the backspace key
    digit: int  # what the tapped key showed: 0..9, KEYPAD_BACK or KEYPAD_BLANK
    expected: int  # the digit the entry needed at that moment, -1 if the entry was already full
    correct: bool  # digit == expected
    x: float  # clientX of the press
    y: float
    pointer_type: str = Field(default="mouse", max_length=16)
    trusted: bool = True


class KeypadRun(BaseModel):
    """One solved (or abandoned) captcha, raw. The record of truth for the keypad."""

    challenge_id: str = Field(max_length=64)
    layout: list[int]
    target: list[int]
    # Rects [x, y, w, h] in client coordinates, keyed by cell index as a string
    # ("0".."5") plus "back" and "target" (the box showing the digits to enter).
    cells: dict[str, list[float]]
    shown_at: float  # t when the layout became visible; the first tap's search starts here
    taps: list[KeypadTap] = Field(max_length=200)
    pointer: list[PointerEvent] = Field(default_factory=list, max_length=MAX_EVENTS)  # whole run, same clock
    env: dict[str, Any] = Field(default_factory=dict)  # probe.js output
    viewport: str = Field(default="", max_length=32)
    completed: bool = True  # the entered digits matched the target


class KeypadIn(BaseModel):
    """Body of POST /api/enroll/keypad (exactly one run per call) and
    POST /api/login/keypad (KEYPAD_STEPUP_RUNS runs, answering a "keypad" decision)."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    runs: list[KeypadRun] = Field(min_length=1, max_length=KEYPAD_STEPUP_RUNS)


class KeypadEnrollOut(BaseModel):
    accepted: bool
    count: int  # accepted keypad runs so far
    target: int  # KEYPAD_ENROLL_RUNS
    enrolled: bool  # a keypad profile now exists
    device_class: DeviceClass = "unknown"
    reasons: list[str] = Field(default_factory=list)


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

    # keypad: the cognitive half of the scrambled keypad (search + reach cadence),
    # scored on any device. keypad_motor: the movement half, only when the
    # device class matches enrollment; advisory.
    name: Literal["keystroke", "pointer", "pointer_neural", "bot", "device", "keypad", "keypad_motor"]
    available: bool = True  # False: not enough data to judge (e.g. no pointer used)
    score: float = 0.0  # larger = less like the owner (or more bot-like)
    threshold: float = 1.0
    flagged: bool = False  # score > threshold, or a hard rule fired
    contributions: list[Contribution] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # human-readable, shown as-is


# "step_up": password right, rhythm right, but the device looks new. Not a login
# yet: the browser is asked for STEP_UP_SAMPLES more typings (POST /api/login/stepup)
# and the median keystroke score of all four decides allow/block.
# "keypad": same situation, but re-typing cannot settle it (touch keyboard, a
# different device class, or a rhythm score too close to its limit): the browser
# is asked for KEYPAD_STEPUP_RUNS scrambled-keypad captchas (POST /api/login/keypad)
# and the keypad's cognitive score decides allow/block.
Decision = Literal["allow", "block", "step_up", "keypad", "retype", "wrong_password", "unknown_user",
                   "not_enrolled"]


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
