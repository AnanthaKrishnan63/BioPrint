"""Keystroke features. OWNER: Agent A (engine).

STUB: pairing and a CMU-layout vector are real enough to run end to end; Agent A
replaces this with the hardened version (phantom keydowns, Shift handling, tests).
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import FeatureVector, KeyEvent, Sample

# Keys that end or navigate the form rather than belong to the password.
EXCLUDED = {"Enter", "NumpadEnter", "Tab"}
CORRECTION = {"Backspace", "Delete"}


@dataclass
class Keystroke:
    code: str
    down: float
    up: float


def pair_keystrokes(events: list[KeyEvent]) -> list[Keystroke]:
    """Match each keydown with its keyup, ordered by keydown time."""
    pending: dict[str, float] = {}
    out: list[Keystroke] = []
    for e in sorted(events, key=lambda e: e.t):
        if e.type == "down":
            pending.setdefault(e.code, e.t)  # a second keydown is auto-repeat or phantom
        elif e.code in pending:
            out.append(Keystroke(e.code, pending.pop(e.code), e.t))
    return sorted(out, key=lambda k: k.down)


def password_keystrokes(sample: Sample) -> list[Keystroke]:
    events = [e for e in sample.keystrokes if e.field == "password"]
    return [k for k in pair_keystrokes(events) if k.code not in EXCLUDED]


def needs_retype(sample: Sample, template_codes: list[str] | None = None) -> str | None:
    """A reason string if this sample cannot be scored, else None.

    Corrections break positional alignment, and a different key sequence (Caps Lock,
    other Shift) cannot be compared against the template.
    """
    ks = password_keystrokes(sample)
    if not ks:
        return "no keystrokes in the password field"
    if any(k.code in CORRECTION for k in ks):
        return "password was corrected mid-way; please type it again without mistakes"
    if template_codes is not None and [k.code for k in ks] != template_codes:
        return "keys pressed differ from enrollment; please type it the usual way"
    return None


def keystroke_vector(keystrokes: list[Keystroke]) -> FeatureVector:
    """CMU layout: H per key, then DD and UD per consecutive pair. Milliseconds."""
    names: list[str] = []
    values: list[float] = []
    label = [f"{k.code}#{i}" for i, k in enumerate(keystrokes)]
    for i, k in enumerate(keystrokes):
        names.append(f"H.{label[i]}")
        values.append(k.up - k.down)
        if i + 1 < len(keystrokes):
            nxt = keystrokes[i + 1]
            names.append(f"DD.{label[i]}.{label[i + 1]}")
            values.append(nxt.down - k.down)
            names.append(f"UD.{label[i]}.{label[i + 1]}")
            values.append(nxt.down - k.up)
    return FeatureVector(names=names, values=values)


def template_codes(vector: FeatureVector) -> list[str]:
    """The key sequence a keystroke vector was built from, recovered from its H.* names."""
    return [n[2:].rsplit("#", 1)[0] for n in vector.names if n.startswith("H.")]
