"""Synthetic samples for tests. Shared by every agent: extend, don't fork."""

from __future__ import annotations

import math
import random

from contracts import KEYPAD_BACK, KEYPAD_BLANK, KEYPAD_DIGITS, KEYPAD_TARGET_LEN

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


# ---------------------------------------------------------------- scrambled keypad

KEY_W, KEY_H, KEY_GAP = 90.0, 70.0, 10.0
PAD_X, PAD_Y = 120.0, 300.0  # top-left of cell 0
KEYPAD_COLS_T = 3  # the generator's grid; contracts.KEYPAD_COLS is the app's


def keypad_cells() -> dict[str, list[float]]:
    """Rects for a 3x2 grid of keys, the backspace key beside it and the box
    showing the digits to enter, all [x, y, w, h] in client coordinates."""
    cells = {}
    for i in range(KEYPAD_COLS_T * 2):
        col, row = i % KEYPAD_COLS_T, i // KEYPAD_COLS_T
        cells[str(i)] = [PAD_X + col * (KEY_W + KEY_GAP), PAD_Y + row * (KEY_H + KEY_GAP), KEY_W, KEY_H]
    cells["back"] = [PAD_X + KEYPAD_COLS_T * (KEY_W + KEY_GAP), PAD_Y, KEY_W, KEY_H]
    cells["target"] = [PAD_X, PAD_Y - 100.0, KEYPAD_COLS_T * KEY_W + 2 * KEY_GAP, 60.0]
    return cells


def _min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def make_run(
    *,
    seed: int = 0,
    device: str = "mouse",        # "mouse" | "touch"
    interval_ms: float = 520.0,   # previous release -> next press: search + reach + decide
    interval_sd: float = 70.0,
    hold_ms: float = 95.0,        # press -> release
    hold_sd: float = 12.0,
    first_extra_ms: float = 300.0,  # the first tap also has to read the target digits
    errors: int = 0,              # wrong digit taps, each followed by a backspace
    layout: list[int] | None = None,
    target: list[int] | None = None,
    land_sd: float = 9.0,         # px of scatter around the key centre
    jitter: float = 0.8,          # px of hand/sensor noise on the captured path
    hover_ms: float = 70.0,       # motionless on the key before pressing (mouse only)
    curve: float = 0.08,          # lateral arc of each reach, as a fraction of its length
    challenge_id: str = "c1",
    shown_at: float = 0.0,
    env: dict | None = None,
    completed: bool | None = None,
    trusted: bool = True,
) -> dict:
    """A JSON-ready contracts.KeypadRun: one solved scrambled-keypad captcha.

    One parameter set is one "person": sampling it twice with different seeds
    gives two captchas by the same person, and changing interval_ms/hold_ms gives
    somebody else. On "mouse" each tap carries a full reach (a curved minimum-jerk
    path, ~20-40 captured moves, a bell-shaped speed profile) so pointer-style
    kinematics are computable; on "touch" only the contact down/up exists, which
    is all a real touchscreen reports.
    """
    rng = random.Random(seed)
    if layout is None:
        layout = list(range(KEYPAD_DIGITS)) + [KEYPAD_BLANK]
        rng.shuffle(layout)
    if target is None:
        target = [rng.randrange(KEYPAD_DIGITS) for _ in range(KEYPAD_TARGET_LEN)]
    cells = keypad_cells()
    where = {d: i for i, d in enumerate(layout) if d >= 0}

    # Tap plan: (cell, digit), with `errors` wrong taps each undone by a backspace.
    wrong_at = set(rng.sample(range(len(target)), min(errors, len(target))))
    plan: list[tuple[int, int]] = []
    for pos, digit in enumerate(target):
        if pos in wrong_at:
            others = [d for d in where if d != digit]
            if others:
                bad = rng.choice(others)
                plan.append((where[bad], bad))
                plan.append((-1, KEYPAD_BACK))
        plan.append((where[digit], digit))

    def point(cell: int) -> tuple[float, float]:
        rx, ry, rw, rh = cells["back" if cell < 0 else str(cell)]
        return (min(max(rx + rw / 2 + rng.gauss(0, land_sd), rx + 2), rx + rw - 2),
                min(max(ry + rh / 2 + rng.gauss(0, land_sd), ry + 2), ry + rh - 2))

    taps: list[dict] = []
    events: list[dict] = []
    entry: list[int] = []
    t_release = float(shown_at)
    cursor = (PAD_X - 140.0, PAD_Y - 180.0)  # where the hand rests before the first reach

    def move(t: float, x: float, y: float) -> None:
        events.append({"type": "move", "t": round(t, 3),
                       "x": round(x + rng.gauss(0, jitter), 2), "y": round(y + rng.gauss(0, jitter), 2),
                       "pointer_type": "mouse", "buttons": 0, "target": None, "trusted": trusted})

    for i, (cell, digit) in enumerate(plan):
        interval = max(90.0, rng.gauss(interval_ms, interval_sd)) + (first_extra_ms if i == 0 else 0.0)
        t_down = t_release + interval
        hold = max(20.0, rng.gauss(hold_ms, hold_sd))
        t_up = t_down + hold
        x, y = point(cell)

        if device == "mouse":
            hov = max(15.0, rng.gauss(hover_ms, 15.0))
            avail = max(40.0, interval - hov)
            move_ms = max(60.0, min(avail * 0.9, avail * 0.55))
            t0 = t_down - hov - move_ms
            dx, dy = x - cursor[0], y - cursor[1]
            dist = math.hypot(dx, dy)
            nx, ny = (-dy / dist, dx / dist) if dist > 1e-6 else (0.0, 0.0)
            n = max(20, min(40, int(move_ms / 8)))
            for k in range(n + 1):
                tau = k / n
                f = _min_jerk(tau)
                arc = curve * dist * math.sin(math.pi * tau)
                move(t0 + tau * move_ms, cursor[0] + dx * f + nx * arc, cursor[1] + dy * f + ny * arc)
            cursor = (x, y)
        for kind, tt in (("down", t_down), ("up", t_up)):
            events.append({"type": kind, "t": round(tt, 3), "x": round(x, 2), "y": round(y, 2),
                           "pointer_type": device, "buttons": 1 if kind == "down" else 0,
                           "target": None, "trusted": trusted})

        expected = target[len(entry)] if len(entry) < len(target) else -1
        taps.append({"t_down": round(t_down, 3), "t_up": round(t_up, 3), "cell": cell, "digit": digit,
                     "expected": expected, "correct": digit == expected,
                     "x": round(x, 2), "y": round(y, 2), "pointer_type": device, "trusted": trusted})
        if digit == KEYPAD_BACK:
            if entry:
                entry.pop()
        elif len(entry) < len(target):
            entry.append(digit)
        t_release = t_up

    if env is None:
        from test_bot import human_env  # late: test_bot imports this module
        if device == "touch":
            env = human_env(ua="Mozilla/5.0 (Android 14; Mobile; rv:155.0) Gecko/155.0 Firefox/155.0",
                            ua_mobile=True, max_touch_points=5, plugins=0, platform="Linux armv8l",
                            webgl_vendor="Qualcomm", webgl_renderer="Adreno (TM) 730",
                            screen_width=412, screen_height=915, inner_width=412, inner_height=800,
                            outer_width=412, outer_height=915, device_memory=4)
        else:
            env = human_env(webgl_vendor="NVIDIA Corporation",
                            webgl_renderer="NVIDIA GeForce RTX 3060/PCIe/SSE2")

    return {
        "challenge_id": challenge_id,
        "layout": list(layout),
        "target": list(target),
        "cells": cells,
        "shown_at": float(shown_at),
        "taps": taps,
        "pointer": events,
        "env": env,
        "viewport": "412x915@3" if device == "touch" else "1920x1050@1",
        "completed": (entry == list(target)) if completed is None else completed,
    }
