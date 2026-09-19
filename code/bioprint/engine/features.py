"""Keystroke features. OWNER: Agent A (engine).

Turns one raw key-event stream into the CMU H/DD/UD vector, and decides when a
sample simply cannot be compared to the template (corrections, a different key
sequence) and the person should be asked to type again.

Three things about real browser streams drive the design here, all measured in
[[big_idea/12 Measurements]] rather than assumed:

  1. Firefox emits **phantom duplicate keydowns** in ~1% of events: the same code
     goes down twice with no release between, the first one a few ms *before* the
     genuine press. Naive pairing measures the hold from the phantom and inflates
     that dwell. We keep the *later* press when the duplicate lands inside
     `PHANTOM_WINDOW_MS`, which is what the kernel log says actually happened.
  2. Keys overlap constantly (the subject held 2+ keys at once 32% of the time),
     so a keyup can arrive after the next keydown, and the last key of the
     password can still be down when Enter submits the form. Order by keydown,
     not by arrival, and tolerate an unreleased key at the end.
  3. A keyup can land in a *different field* from its keydown (Tab/Enter moves
     focus between the two events). Pair over the whole stream and attribute the
     keystroke to the field its **keydown** happened in, otherwise the last
     character of a fast typist disappears and the sample is rejected forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import FeatureVector, KeyEvent, Sample

# Keys that end or navigate the form rather than belong to the password.
EXCLUDED = {"Enter", "NumpadEnter", "Tab", "Escape"}

# Keys that mean the text was edited, so position i of the sample is no longer
# position i of the template. Arrows/Home/End are here for the same reason as
# Backspace: they move the caret, and what follows is inserted somewhere else.
CORRECTION = {"Backspace", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"}

# Left/right variants of the same modifier are the same intention. A person who
# reaches for the other Shift between enrollment and login must not be locked out,
# so the *canonical* code is what the template records and compares.
MODIFIER_PAIRS = {
    "ShiftLeft": "Shift", "ShiftRight": "Shift",
    "ControlLeft": "Control", "ControlRight": "Control",
    "AltLeft": "Alt", "AltRight": "Alt",
    "MetaLeft": "Meta", "MetaRight": "Meta",
    "OSLeft": "Meta", "OSRight": "Meta",  # older Firefox spelling of the Meta keys
}
MODIFIERS = set(MODIFIER_PAIRS) | set(MODIFIER_PAIRS.values())

# A second keydown for a key that is already down, this soon after the first, is a
# phantom (measured: ~5 ms early, all four occurrences at points where keys
# overlapped inside the browser's 1 ms tick). Later than this it is treated as OS
# auto-repeat that escaped `event.repeat` filtering, and the first press wins.
PHANTOM_WINDOW_MS = 50.0

# Modifiers are NOT keystrokes in the positional vector. Reasons, in order of weight:
#   * CMU compatibility. `DSL-StrongPasswordData.csv` has 11 keystroke columns for
#     `.tie5Roanl` + Return, and the capital R appears as the single keystroke
#     `Shift.r` — the Shift key itself has no H/DD/UD column. Dropping modifiers
#     makes our live vector exactly their layout, so the EER measured on their
#     51 subjects is a statement about *this* feature design.
#   * Alignment robustness. Holding Shift across two capitals on one login and
#     re-pressing it on the next changes the key *sequence*, which under a
#     positional template means "please type it again" forever. The shifted
#     character's own key code is unchanged either way, so nothing about which
#     characters were typed is lost.
# The cost is real and worth stating: Shift dwell and the Shift->letter lead time
# are discriminative, and we give them up. Revisit once there is live data to
# measure the trade on; flip this to False and the codes stay canonicalised.
INCLUDE_MODIFIERS = False


@dataclass
class Keystroke:
    code: str
    down: float
    up: float


def canonical_code(code: str) -> str:
    """Template spelling of a physical key: left/right modifiers collapse to one."""
    return MODIFIER_PAIRS.get(code, code)


def _paired(events: list[KeyEvent]) -> list[tuple[KeyEvent, Keystroke]]:
    """Every press with the keydown event it came from, ordered by press time.

    Tolerates, in this order: phantom duplicate keydowns, auto-repeat, keyups with
    no keydown (the press happened before the form was reset, or in another
    window), and keys still held when the sample was taken.
    """
    # Sort by time, ties broken by arrival order: within one 1 ms tick the order
    # events were delivered in is the best evidence of what happened first.
    order = sorted(range(len(events)), key=lambda i: (events[i].t, i))
    pending: dict[str, tuple[int, KeyEvent, float]] = {}  # code -> (seq, keydown, press time)
    out: list[tuple[int, KeyEvent, Keystroke]] = []
    last_t = max((e.t for e in events), default=0.0)

    for seq, i in enumerate(order):
        e = events[i]
        if e.type == "down":
            held = pending.get(e.code)
            if held is None:
                pending[e.code] = (seq, e, e.t)
            elif e.t - held[2] <= PHANTOM_WINDOW_MS:
                # Phantom: the genuine press is the *second* one. Keep the original
                # position in the stream but take the later, true press time, so
                # both this key's dwell and the flight time into it are right.
                pending[e.code] = (held[0], held[1], e.t)
            # else: auto-repeat (or a keyup we never saw). The first press stands.
        else:
            held = pending.pop(e.code, None)
            if held is not None and e.t >= held[2]:
                out.append((held[0], held[1], Keystroke(canonical_code(e.code), held[2], e.t)))

    # Still held when the sample was taken. Common and benign: submitting with
    # Enter serialises the sample while the last character is often still down.
    # Closing it at the last event we saw makes the dwell a lower bound, which
    # keeps the keystroke (and the whole sample) instead of throwing it away.
    for seq, down, t in pending.values():
        out.append((seq, down, Keystroke(canonical_code(down.code), t, max(t, last_t))))

    out.sort(key=lambda r: (r[2].down, r[0]))
    return [(down, k) for _, down, k in out]


def pair_keystrokes(events: list[KeyEvent]) -> list[Keystroke]:
    """Match each keydown with its keyup, ordered by keydown time."""
    return [k for _, k in _paired(events)]


def password_keystrokes(sample: Sample) -> list[Keystroke]:
    """The presses that make up the password, in press order, template spelling."""
    out = []
    for down, k in _paired(sample.keystrokes):
        # The keydown decides the field: a keyup can arrive after focus moved on.
        if down.field != "password" or k.code in EXCLUDED:
            continue
        if not INCLUDE_MODIFIERS and k.code in MODIFIERS:
            continue
        out.append(k)
    return out


def is_virtual_keyboard(sample: Sample) -> bool:
    """Mobile keyboards (Android Firefox/Chrome IMEs) send keydown/keyup with an empty
    or 'Unidentified' code and a 0 ms hold: a real person, but no desktop rhythm."""
    downs = [e for e in sample.keystrokes if e.type == "down" and e.field == "password"]
    return bool(downs) and all(e.code in ("", "Unidentified") for e in downs)


def needs_retype(sample: Sample, template_codes: list[str] | None = None) -> str | None:
    """A reason string if this sample cannot be scored, else None.

    Corrections break positional alignment, and a different key sequence (Caps Lock,
    a different layout, a mistyped-then-fixed password) cannot be compared against
    the template. `meta.had_paste` is deliberately *not* checked here: it is set by
    a paste anywhere in the form, including the username, and pasting the password
    leaves no key events at all, which the first check below already catches.
    """
    # Pasting fires no key events for the text, so there is no rhythm to compare.
    # Checked first: otherwise the Ctrl+V keys read as "keys differ from enrollment".
    if sample.meta.had_paste:
        return "the password was pasted; please type it, we compare the rhythm"
    if is_virtual_keyboard(sample):
        return ("typed on a touch keyboard, which reports no key timing; BioPrint profiles are "
                "per device, so enrol on this phone to sign in from it")
    ks = password_keystrokes(sample)
    if not ks:
        return "no keystrokes in the password field"
    if any(k.code in CORRECTION for k in ks):
        return "password was corrected mid-way; please type it again without mistakes"
    if template_codes is not None:
        want = [canonical_code(c) for c in template_codes]
        if [k.code for k in ks] != want:
            return "keys pressed differ from enrollment; please type it the usual way"
    return None


def keystroke_vector(keystrokes: list[Keystroke]) -> FeatureVector:
    """CMU layout: H per key, then DD and UD per consecutive pair. Milliseconds.

    H  = dwell, press to release of one key.
    DD = press of key i to press of key i+1 (always positive).
    UD = release of key i to press of key i+1; negative when the keys overlap,
         which for this subject happens about a third of the time.
    The `#i` suffix keeps repeated characters apart ("H.KeyA#0" vs "H.KeyA#3") and
    makes the name order a positional template the scorer can rely on.
    """
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


def correction_count(sample: Sample) -> int:
    """Backspace/Delete/caret presses in the password field: how often a person
    notices and fixes a slip. Kept per sample so correction habits can be studied
    (and, later, scored) even though a corrected sample is never aligned."""
    return sum(1 for e in sample.keystrokes
               if e.type == "down" and e.field == "password" and e.code in CORRECTION)
