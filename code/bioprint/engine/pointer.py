"""Pointer features. OWNER: Agent C (pointer).

One fixed-length vector describing the final reach-and-click on the button that
submitted this sample (meta.targets["submit"]). The server fits engine.scorer on
it exactly as it does for keystrokes.

Design rules
- Only the LAST movement episode before the click is used: the hand leaves the
  keyboard, moves to the button, clicks. Anything earlier is page wandering.
- Where the cursor happened to start differs between tries, so distance-dependent
  quantities are normalised (Fitts's index of difficulty, speed-profile ratios,
  deviations as a fraction of distance, landing point as a fraction of the button).
- Moves are resampled to a fixed 10 ms grid before any speed is computed, so the
  vector does not depend on the mouse's polling rate or on coalesced-event density.
- Every feature is continuous and rarely constant, so a per-feature spread fitted
  on 10 tries is never ~0 (an always-zero feature makes scaled Manhattan explode).
- Times are capped so one distracted try cannot dominate the scaled-Manhattan fit.

Features (all prefixed "pointer."):
  key_to_move_ms   last key released -> approach starts ("hand moves keyboard -> mouse")
  ms_per_bit       movement time / log2(distance/width + 1): Fitts slope, start-independent
  path_efficiency  straight-line distance / travelled path (1 = perfectly straight)
  curvature        max perpendicular deviation from the straight line / distance
  peak_mean_ratio  peak speed / mean speed: shape of the velocity profile, not its scale
  time_to_peak     fraction of the approach elapsed when speed peaks (bell = ~0.4-0.5)
  speed_variation  total variation of speed after the peak / peak: 1 = one smooth
                   deceleration, each corrective sub-movement adds ~2x its relative height
  homing_frac      fraction of the approach spent after speed last falls below 20% of peak
  overshoot        furthest progress along the start->centre axis, minus the distance, / distance
                   (> 0: went past the centre; < 0: never reached it)
  land_dx          click x offset from button centre / button width   (-0.5 .. 0.5 inside)
  land_dy          click y offset from button centre / button height
  hover_ms         final arrival on the button -> button pressed (hesitation, corrections included)
  settle_ms        last movement -> button pressed (motionless pause before the click)
  hold_ms          button pressed -> released (click dwell)
"""

from __future__ import annotations

import math

import numpy as np

from contracts import FeatureVector, PointerEvent, Sample

NAMES = [
    "pointer.key_to_move_ms",
    "pointer.ms_per_bit",
    "pointer.path_efficiency",
    "pointer.curvature",
    "pointer.peak_mean_ratio",
    "pointer.time_to_peak",
    "pointer.speed_variation",
    "pointer.homing_frac",
    "pointer.overshoot",
    "pointer.land_dx",
    "pointer.land_dy",
    "pointer.hover_ms",
    "pointer.settle_ms",
    "pointer.hold_ms",
]

EPISODE_GAP_MS = 250.0  # no move for this long = the pointer was at rest; an episode starts after it
MIN_MOVES = 5  # fewer moves than this in the approach: nothing to measure
GRID_MS = 10.0  # resampling step for the speed profile
SMOOTH = 3  # moving-average window on the grid (30 ms)
TARGET_MARGIN = 2.0  # px of slack when asking "was this inside the button?"
CAP_MS = 5000.0  # cap on every duration feature


def _inside(x: float, y: float, rect: list[float], margin: float = TARGET_MARGIN) -> bool:
    rx, ry, rw, rh = rect
    return rx - margin <= x <= rx + rw + margin and ry - margin <= y <= ry + rh + margin


def _click(events: list[PointerEvent], rect: list[float]) -> tuple[PointerEvent, PointerEvent | None] | None:
    """The last press on the submit button, and its release (if recorded)."""
    for i in range(len(events) - 1, -1, -1):
        e = events[i]
        if e.type == "down" and _inside(e.x, e.y, rect):
            up = next((u for u in events[i + 1:] if u.type == "up"), None)
            return e, up
    return None


def _episode(events: list[PointerEvent], t_down: float) -> list[PointerEvent]:
    """Moves of the last continuous movement before the press."""
    moves = [e for e in events if e.type == "move" and e.t <= t_down]
    if not moves:
        return []
    start = len(moves) - 1
    while start > 0 and moves[start].t - moves[start - 1].t <= EPISODE_GAP_MS:
        start -= 1
    return moves[start:]


def _speed_profile(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Smoothed speed (px/ms) on a fixed GRID_MS grid."""
    grid = np.arange(t[0], t[-1] + GRID_MS / 2, GRID_MS)
    if len(grid) < 3:
        grid = np.linspace(t[0], t[-1], 3)
    gx = np.interp(grid, t, xy[:, 0])
    gy = np.interp(grid, t, xy[:, 1])
    v = np.hypot(np.diff(gx), np.diff(gy)) / np.diff(grid)
    if len(v) >= SMOOTH:
        v = np.convolve(v, np.ones(SMOOTH) / SMOOTH, mode="same")
    return v


def _cap(ms: float) -> float:
    return float(min(max(ms, 0.0), CAP_MS))


def pointer_vector(sample: Sample) -> FeatureVector | None:
    """Fixed-length vector describing how the pointer reached and clicked the submit
    button. None when the sample was submitted with Enter, the button's rect is
    missing, or there is too little pointer movement to measure."""
    meta = sample.meta
    rect = meta.targets.get("submit")
    if meta.submit_via == "enter" or not rect or len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
        return None
    events = sorted(sample.pointer, key=lambda e: e.t)
    click = _click(events, rect)
    if click is None:
        return None
    down, up = click
    if up is None:
        return None  # a click that never released cannot have submitted anything
    moves = _episode(events, down.t)
    if len(moves) < MIN_MOVES:
        return None

    rx, ry, rw, rh = rect
    cx, cy = rx + rw / 2, ry + rh / 2
    t = np.array([m.t for m in moves] + [down.t], dtype=float)
    xy = np.array([[m.x, m.y] for m in moves] + [[down.x, down.y]], dtype=float)
    # Timestamps can tie (coalesced events); keep them strictly increasing for interp.
    t = t + np.arange(len(t)) * 1e-6

    start = xy[0]
    dist = float(math.hypot(cx - start[0], cy - start[1]))
    if _inside(start[0], start[1], rect, 0.0) or dist < 0.5 * min(rw, rh):
        return None  # already on the button: there was no approach

    # Arrival: first point of the final run of points inside the button.
    inside = np.array([_inside(px, py, rect) for px, py in xy])
    i_arr = len(xy) - 1
    while i_arr > 0 and inside[i_arr - 1]:
        i_arr -= 1
    t_arr = float(t[i_arr])
    t_end = float(t[-2])  # last recorded move: the end of the movement proper
    approach = t_end - float(t[0])
    if approach <= 0:
        return None

    # Last key released before the hand started moving.
    ups = [k.t for k in sample.keystrokes if k.type == "up" and k.t <= t[0]]
    key_to_move = t[0] - max(ups) if ups else t[0]

    # Fitts: effective width along the approach direction.
    ux, uy = (cx - start[0]) / dist, (cy - start[1]) / dist
    width = min(rw / abs(ux) if abs(ux) > 1e-9 else math.inf, rh / abs(uy) if abs(uy) > 1e-9 else math.inf)
    bits = math.log2(dist / width + 1.0)

    # Path shape, over the whole episode up to the press.
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    path = float(seg.sum())
    click_pt = xy[-1]
    straight = float(np.hypot(*(click_pt - start)))
    efficiency = straight / path if path > 0 else 1.0
    rel = xy - start
    perp = np.abs(rel[:, 0] * uy - rel[:, 1] * ux)
    curvature = float(perp.max()) / dist
    along = rel[:, 0] * ux + rel[:, 1] * uy
    overshoot = (float(along.max()) - dist) / dist

    # Velocity profile from movement start to the last move (in-button corrections
    # included; the motionless hover before the press is not).
    v = _speed_profile(t[:-1], xy[:-1])
    mean_v = float(v.mean())
    i_peak = int(np.argmax(v))
    peak_mean = float(v[i_peak]) / mean_v if mean_v > 0 else 1.0
    time_to_peak = (i_peak + 0.5) / len(v)
    variation = float(np.abs(np.diff(v[i_peak:])).sum()) / float(v[i_peak]) if v[i_peak] > 0 else 0.0
    below = np.nonzero(v[i_peak:] >= 0.2 * v[i_peak])[0]
    homing = 1.0 - (i_peak + int(below[-1]) + 1) / len(v) if len(below) else 0.0

    values = [
        _cap(key_to_move),
        min(approach / bits, CAP_MS),
        efficiency,
        curvature,
        peak_mean,
        time_to_peak,
        variation,
        homing,
        overshoot,
        (down.x - cx) / rw,
        (down.y - cy) / rh,
        _cap(down.t - t_arr),
        _cap(down.t - t_end),
        _cap(up.t - down.t),
    ]
    values = [float(x) if math.isfinite(x) else 0.0 for x in values]
    return FeatureVector(names=list(NAMES), values=values)
