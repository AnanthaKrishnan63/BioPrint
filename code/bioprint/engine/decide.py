"""The one place signals become a decision. OWNER: main session (tuned in phase 2).

Signals stay separate all the way here; this function reads them side by side and
explains itself. BioPrint requires a hard block, with no OTP or friction fallback.

The device x keystroke table (approved by the project lead):

                     keystroke passes    keystroke fails
    same device      allow               block
    different device step_up             block

"step_up" is not a second factor: the browser is asked to type the password a
few more times and the median keystroke score of all the samples decides. A bot
flag blocks before anything else. Pointer stays advisory. A device signal that
is unavailable (no probe, old account) counts as "same device".
"""

from __future__ import annotations

from contracts import Decision, SignalResult


def decide(signals: list[SignalResult], after_step_up: bool = False) -> tuple[Decision, list[str]]:
    """`after_step_up`: the extra samples have been collected, so a new device can
    no longer ask for more; it is reported and the keystroke verdict stands."""
    by = {s.name: s for s in signals}
    reasons: list[str] = []

    bot = by.get("bot")
    if bot and bot.flagged:
        return "block", ["automated or replayed input"] + bot.reasons

    dv = by.get("device")
    new_device = bool(dv and dv.available and dv.flagged)

    ks = by.get("keystroke")
    if ks and ks.available and ks.flagged:
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

    if new_device and not after_step_up:
        return "step_up", ["typing rhythm matches, but this looks like a new device: "
                           "a few more typings will settle it"] + dv.reasons + reasons

    if new_device:
        reasons.append("this looks like a different device from enrollment (advisory)")
        reasons += dv.reasons

    return "allow", reasons or ["typing rhythm matches the account owner"]
