"""Device axis: is this the browser the profile was enrolled on? OWNER: device agent.

Advisory, never a block on its own: the brief's demo is an impostor on the
owner's own laptop, where this signal correctly says "same device". It exists
so the dashboard can show the disagreement pattern (behaviour off, device same
= someone at the owner's machine; both off = a different person elsewhere).

STUB: always "same device". The device agent replaces it.
"""

from __future__ import annotations

from contracts import SignalResult


def check(env: dict, enrolled_envs: list[dict]) -> SignalResult:
    """`env` is this attempt's probe output; `enrolled_envs` the probe outputs of
    the accepted enrollment samples (may be empty for old accounts)."""
    return SignalResult(name="device", available=bool(enrolled_envs), score=0.0, threshold=1.0,
                        reasons=["same device as enrollment"] if enrolled_envs else ["no enrollment environment"])
