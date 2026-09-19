"""The one place signals become a decision. OWNER: main session (tuned in phase 2).

Signals stay separate all the way here; this function reads them side by side and
explains itself. BioPrint requires a hard block, with no OTP or friction fallback.
"""

from __future__ import annotations

from contracts import Decision, SignalResult


def decide(signals: list[SignalResult]) -> tuple[Decision, list[str]]:
    by = {s.name: s for s in signals}
    reasons: list[str] = []

    bot = by.get("bot")
    if bot and bot.flagged:
        return "block", ["automated or replayed input"] + bot.reasons

    ks = by.get("keystroke")
    if ks and ks.available and ks.flagged:
        reasons.append(f"typing rhythm is unlike the account owner ({ks.score:.1f} vs limit {ks.threshold:.1f})")
        reasons += ks.reasons
        return "block", reasons

    pt = by.get("pointer")
    # Pointer alone does not block yet: one behavioural channel with ~10 enrollment
    # samples is too noisy to veto the keystroke verdict. Revisit with phase-2 data.
    if pt and pt.available and pt.flagged:
        reasons.append("pointer movement was unusual (not decisive on its own)")

    dv = by.get("device")
    if dv and dv.available and dv.flagged:
        reasons.append("this looks like a different device from enrollment (advisory)")
        reasons += dv.reasons

    return "allow", reasons or ["typing rhythm matches the account owner"]
