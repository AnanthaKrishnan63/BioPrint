"""The one place signals become a decision. OWNER: main session (tuned in phase 2).

Signals stay separate all the way here; this function reads them side by side and
explains itself. BioPrint requires a hard block, with no OTP or friction fallback.

The device x keystroke table (approved by the project lead):

                     keystroke passes    keystroke fails
    same device      allow               block
    different device step_up / keypad    block

Neither step is a second factor: both ask for more of the same behavioural
evidence. "step_up" asks the browser to type the password a few more times and
the median keystroke score decides. "keypad" asks for two scrambled-keypad
captchas and the keypad's cognitive score decides. Which one (approved routing):

    new device and ...                                        route
    the device class differs (phone <-> laptop)               keypad
    no key timing at all (touch keyboard)                     keypad
    rhythm passes but is near its limit (>= BORDERLINE)       keypad
    the typing step-up's median is still near the limit       keypad
    otherwise (same class, rhythm passes comfortably)         step_up

The keypad routes need a keypad profile; without one, the typing step-up is the
only option, and a touch keyboard stays "retype" (handled by the server).

A bot flag blocks before anything else. Pointer and keypad_motor stay advisory.
A device signal that is unavailable (no probe, old account) counts as "same device".
"""

from __future__ import annotations

from contracts import Decision, SignalResult

BORDERLINE = 0.8  # keystroke score / threshold at or above this is "near the limit"


def decide(signals: list[SignalResult], after_step_up: bool = False, keypad_enrolled: bool = False,
           new_class: bool = False) -> tuple[Decision, list[str]]:
    """`after_step_up`: the extra typings have been collected, so a new device can
    no longer ask for more of them (it may still ask for the keypad).
    `keypad_enrolled`: the account has a keypad profile, so "keypad" is possible.
    `new_class`: the browser's device class (touch vs mouse) differs from the one
    the keypad was enrolled on."""
    by = {s.name: s for s in signals}
    reasons: list[str] = []

    bot = by.get("bot")
    if bot and bot.flagged:
        return "block", ["automated or replayed input"] + bot.reasons

    dv = by.get("device")
    new_device = bool(dv and dv.available and dv.flagged)

    ks = by.get("keystroke")
    ks_available = bool(ks and ks.available)
    if ks_available and ks.flagged:
        reasons.append(f"typing rhythm is unlike the account owner ({ks.score:.1f} vs limit {ks.threshold:.1f})")
        reasons += ks.reasons
        if new_device:
            reasons.append("and this looks like a different device from enrollment")
            reasons += dv.reasons
        return "block", reasons

    pt = by.get("pointer")
    # Pointer alone does not block yet: one behavioural channel with ~10 enrollment
    # samples is too noisy to veto the keystroke verdict. Revisit with phase-2 data.
    if pt and pt.available and pt.flagged:
        reasons.append("pointer movement was unusual (not decisive on its own)")

    if new_device:
        borderline = ks_available and ks.threshold > 0 and ks.score / ks.threshold >= BORDERLINE
        # After the typing step-up the keystroke signal is the median of four
        # samples, so "borderline" there means more typing did not settle it.
        if keypad_enrolled and (not ks_available or new_class or borderline):
            why = ("no key timing on this keyboard" if not ks_available
                   else "a different kind of device" if new_class
                   else f"the typing rhythm passes but only just ({ks.score:.1f} vs limit {ks.threshold:.1f})")
            return "keypad", [f"this looks like a new device and {why}: "
                              "two quick keypad checks will settle it"] + dv.reasons + reasons
        if ks_available and not after_step_up:
            return "step_up", ["typing rhythm matches, but this looks like a new device: "
                               "a few more typings will settle it"] + dv.reasons + reasons
        reasons.append("this looks like a different device from enrollment (advisory)")
        reasons += dv.reasons

    if not ks_available:
        # Only reachable with no keystrokes and no new device, which the server
        # answers with "retype" before calling here; never allow on nothing.
        return "block", ["no typing rhythm to compare"] + reasons

    return "allow", reasons or ["typing rhythm matches the account owner"]


def decide_keypad(signals: list[SignalResult]) -> tuple[Decision, list[str]]:
    """After the keypad captchas: the cognitive keypad score decides, bot first."""
    by = {s.name: s for s in signals}
    bot = by.get("bot")
    if bot and bot.flagged:
        return "block", ["automated or replayed input"] + bot.reasons

    kp = by.get("keypad")
    if not kp or not kp.available:
        return "block", ["the keypad checks could not be scored"] + (kp.reasons if kp else [])
    reasons: list[str] = []
    if kp.flagged:
        reasons.append(f"keypad behaviour is unlike the account owner ({kp.score:.2f} vs limit {kp.threshold:.2f})")
        reasons += kp.reasons
        return "block", reasons

    reasons.append(f"keypad behaviour matches the account owner ({kp.score:.2f} vs limit {kp.threshold:.2f})")
    km = by.get("keypad_motor")
    if km and km.available and km.flagged:
        reasons.append("keypad movement was unusual (advisory)")
        reasons += km.reasons
    dv = by.get("device")
    if dv and dv.available and dv.flagged:
        reasons.append("signed in from a new device on keypad behaviour alone")
    return "allow", reasons
