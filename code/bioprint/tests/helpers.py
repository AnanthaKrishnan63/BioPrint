"""Synthetic samples for tests. Shared by every agent: extend, don't fork."""

from __future__ import annotations

import random

PASSWORD = "tie5"  # typed as the codes below
CODES = ["KeyT", "KeyI", "KeyE", "Digit5"]


def make_sample(
    codes: list[str] = CODES,
    hold: float = 100.0,
    gap: float = 150.0,  # keydown to next keydown
    jitter: float = 10.0,
    seed: int | None = None,
    trusted: bool = True,
    env: dict | None = None,
    start: float = 500.0,
) -> dict:
    """A JSON-ready contracts.Sample: `codes` typed into the password field."""
    rng = random.Random(seed)
    events = []
    t = start
    for c in codes:
        h = max(20.0, rng.gauss(hold, jitter))
        events.append({"code": c, "type": "down", "t": t, "field": "password", "trusted": trusted})
        events.append({"code": c, "type": "up", "t": t + h, "field": "password", "trusted": trusted})
        t += max(30.0, rng.gauss(gap, jitter))
    return {"keystrokes": events, "pointer": [], "env": env or {}, "meta": {"submit_via": "enter"}}
