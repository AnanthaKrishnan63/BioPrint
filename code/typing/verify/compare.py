#!/usr/bin/env python3
"""Compare the browser's key timings against the kernel's.

The kernel stamps each key in the input driver; the browser stamps it after X11
and its own event plumbing. If the browser is faithful, the two streams differ by
a constant offset and nothing else. A constant offset is harmless -- it cancels
out of every interval. Jitter does not cancel, and eats straight into dwell and
flight measurements.

    python compare.py --kernel kernel.jsonl            # newest saved session
    python compare.py --kernel kernel.jsonl --session 3
    python compare.py --kernel kernel.jsonl --browser session-123.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import sqlite3
import statistics as st
from math import gcd
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DB = HERE.parent / "sessions.db"


# --------------------------------------------------------------------- loading

def load_kernel(path: Path) -> list[dict]:
    """Kernel log: seconds since the Unix epoch. Rebased to ms from the first event."""
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events:
        raise SystemExit(f"{path} is empty. Did the logger record anything?")
    t0 = events[0]["t"]
    return [{"code": e["code"], "type": e["type"], "t": (e["t"] - t0) * 1000.0}
            for e in events]


def load_browser_db(db: Path, session_id: int | None) -> tuple[int, list[dict]]:
    if not db.exists():
        raise SystemExit(f"No database at {db}. Save a session from the page first.")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    if session_id is None:
        row = con.execute("SELECT MAX(id) AS id FROM sessions").fetchone()
        session_id = row["id"]
        if session_id is None:
            raise SystemExit("No sessions saved yet. Type in the page and press Save.")
    rows = con.execute(
        "SELECT code, type, t_ms FROM keystrokes WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    con.close()
    if not rows:
        raise SystemExit(f"Session {session_id} has no key events.")
    t0 = rows[0]["t_ms"]
    return session_id, [{"code": r["code"], "type": r["type"], "t": r["t_ms"] - t0}
                        for r in rows]


def load_browser_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    events = payload["events"] if isinstance(payload, dict) else payload
    t0 = events[0]["t"]
    return [{"code": e["code"], "type": e["type"], "t": e["t"] - t0} for e in events]


# ------------------------------------------------------- shared derivations
# One implementation, applied to both streams, so a bug cannot favour either.

def pair_keystrokes(events: list[dict]) -> list[dict]:
    pending: dict[str, float] = {}
    pairs = []
    for e in events:
        if e["type"] == "down":
            pending.setdefault(e["code"], e["t"])
        elif e["code"] in pending:
            down = pending.pop(e["code"])
            pairs.append({"code": e["code"], "downT": down, "upT": e["t"],
                          "dwell": e["t"] - down})
    pairs.sort(key=lambda p: p["downT"])
    return pairs


def flight_times(pairs: list[dict]) -> list[float]:
    return [pairs[i + 1]["downT"] - pairs[i]["upT"] for i in range(len(pairs) - 1)]


def resolution_ms(events: list[dict]) -> float | None:
    """Largest tick every timestamp is a multiple of."""
    stamps = sorted({e["t"] for e in events})
    if len(stamps) < 2:
        return None
    scale = 10 ** 6
    g = 0
    for s in stamps[1:]:
        g = gcd(g, round((s - stamps[0]) * scale))
    return g / scale if g else None


# ------------------------------------------------------------------ reporting

def p95_abs(xs: list[float]) -> float:
    """95th percentile of the absolute error."""
    if not xs:
        return 0.0
    absxs = sorted(abs(x) for x in xs)
    idx = min(len(absxs) - 1, max(0, round(0.95 * len(absxs)) - 1))
    return absxs[idx]


def describe(xs: list[float], unit: str = "ms") -> str:
    if not xs:
        return "  (no data)"
    p95 = p95_abs(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return (f"  n={len(xs)}  mean={st.mean(xs):+.3f}{unit}  median={st.median(xs):+.3f}{unit}\n"
            f"  sd={sd:.3f}{unit}  min={min(xs):+.3f}{unit}  max={max(xs):+.3f}{unit}"
            f"  p95|error|={p95:.3f}{unit}")


def _token(e: dict) -> str:
    return f"{e['code']}:{e['type']}"


def _walk(a: list[dict], b: list[dict], offset: float, tol: float) -> list[tuple[int, int]]:
    """Single pass: pair events that agree on key AND land close together in time."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        at, bt = a[i]["t"] + offset, b[j]["t"]
        if _token(a[i]) == _token(b[j]) and abs(bt - at) <= tol:
            out.append((i, j))
            i += 1
            j += 1
        elif at <= bt:
            i += 1          # this kernel event has no partner; the browser lost it
        else:
            j += 1          # this browser event has no partner
    return out


def align(a: list[dict], b: list[dict], tol: float = 50.0) -> tuple[list[tuple[int, int]], float]:
    """Pair up the two streams.

    Two things make this harder than it looks.

    Sequence-shape matching (difflib) is wrong: typing repeats the same keys over
    and over, so dropping one event lets a shape matcher lock onto the wrong
    repetition and report dozens of phantom losses. Time is the reliable anchor --
    keys are ~100 ms apart, while the error we are hunting is a few ms.

    But the clocks start at different moments and the kernel log usually runs
    longer at both ends (you start it before opening the page and stop it after
    saving). So the offset between the streams is unknown and can be large.

    We recover it by voting: every pair of events that agree on key implies a
    candidate offset. Genuinely corresponding pairs all imply the *same* offset,
    so the true value shows up as a spike in the histogram, while coincidental
    matches scatter. We take the strongest few spikes, walk the streams with each,
    and keep whichever pairs up the most events.
    """
    votes: Counter = Counter()
    BIN = 2.0  # ms
    by_token: dict[str, list[float]] = {}
    for e in b:
        by_token.setdefault(_token(e), []).append(e["t"])
    for e in a:
        for bt in by_token.get(_token(e), ()):
            votes[round((bt - e["t"]) / BIN)] += 1

    candidates = [bucket * BIN for bucket, _ in votes.most_common(6)] or [0.0]

    best: list[tuple[int, int]] = []
    best_offset = 0.0
    for offset in candidates:
        matched = _walk(a, b, offset, tol)
        if len(matched) > len(best):
            best, best_offset = matched, offset

    # Refine: the winning offset is only accurate to a bin, so re-centre it on the
    # median of the matches it found and walk once more.
    for _ in range(2):
        if not best:
            break
        refined = best_offset + st.median([b[j]["t"] - a[i]["t"] - best_offset
                                           for i, j in best])
        matched = _walk(a, b, refined, tol)
        if len(matched) >= len(best):
            best, best_offset = matched, refined
        else:
            break

    return best, best_offset


def gap_report(stream: list[dict], matched_idx: set[int], name: str) -> str:
    """Split unmatched events into before / during / after the overlapping period.

    Only the middle group means anything. Events before the first match and after
    the last are simply the period when only one recorder was running.
    """
    if not matched_idx:
        return f"  {name}: no overlap found"
    lo, hi = min(matched_idx), max(matched_idx)
    before = lo
    after = len(stream) - 1 - hi
    during = sum(1 for i in range(lo, hi + 1) if i not in matched_idx)
    return (f"  {name}: {before} before overlap, {during} DURING overlap, "
            f"{after} after overlap")


def slope_per_second(times_ms: list[float], deltas_ms: list[float]) -> float:
    """Least-squares drift: ms of error gained per second elapsed."""
    n = len(times_ms)
    if n < 3:
        return 0.0
    mx, my = st.mean(times_ms), st.mean(deltas_ms)
    num = sum((x - mx) * (y - my) for x, y in zip(times_ms, deltas_ms))
    den = sum((x - mx) ** 2 for x in times_ms)
    return (num / den) * 1000.0 if den else 0.0


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", type=Path, required=True, help="JSONL from evdev_logger.py")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--session", type=int, default=None, help="session id; newest if omitted")
    ap.add_argument("--browser", type=Path, default=None, help="exported session JSON instead of the db")
    args = ap.parse_args()

    kernel = load_kernel(args.kernel)
    if args.browser:
        browser, label = load_browser_json(args.browser), str(args.browser)
    else:
        sid, browser = load_browser_db(args.db, args.session)
        label = f"session #{sid}"

    print("=" * 62)
    print("  Browser clock vs kernel clock")
    print("=" * 62)

    rule("Streams")
    print(f"  kernel  : {len(kernel):5d} events over {kernel[-1]['t'] / 1000:7.2f}s  ({args.kernel})")
    print(f"  browser : {len(browser):5d} events over {browser[-1]['t'] / 1000:7.2f}s  ({label})")

    # -------------------------------------------------- 1. did anything go missing
    matches, offset = align(kernel, browser)

    rule("1. Sequence integrity")
    k_idx = {i for i, _ in matches}
    b_idx = {j for _, j in matches}
    print(f"  matched events        : {len(matches)}")
    print(gap_report(kernel, k_idx, "kernel unmatched "))
    print(gap_report(browser, b_idx, "browser unmatched"))
    print(f"  constant clock offset : {offset:+.1f} ms   (harmless; cancels out of every interval)")
    print("\n  Only the DURING figures matter. Events outside the overlap are just the")
    print("  time when one recorder was running and the other was not.")
    if len(matches) < 10:
        raise SystemExit("\n  Too few matched events to analyse. Were both recording at once?")

    # ------------------------------------------- 2. per-event error after rebasing
    # Both streams are already relative to their own first event, so any remaining
    # difference is the browser mis-timing the key rather than a clock offset.
    k_rel = [kernel[i]["t"] - kernel[matches[0][0]]["t"] for i, _ in matches]
    b_rel = [browser[j]["t"] - browser[matches[0][1]]["t"] for _, j in matches]
    deltas = [b - k for k, b in zip(k_rel, b_rel)]

    rule("2. Per-event timing error (browser minus kernel)")
    print(describe(deltas))
    drift = slope_per_second(k_rel, deltas)
    print(f"\n  drift: {drift:+.4f} ms per second elapsed"
          f"  ({'clock rates differ' if abs(drift) > 0.05 else 'clock rates agree'})")

    # -------------------------------------------------------- 3. dwell and flight
    # Derive keystrokes ONLY from events present in both streams. If the browser
    # lost a keyup, pairing its stream alone would marry that keydown to some
    # later release of the same key and invent a dwell of over a second -- an
    # artefact of the drop, not a timing error. Dropping the whole keystroke from
    # this comparison keeps the two questions separate: section 1 counts losses,
    # section 3 measures accuracy on what survived.
    k_matched = [kernel[i] for i, _ in matches]
    b_matched = [browser[j] for _, j in matches]
    kp, bp = pair_keystrokes(k_matched), pair_keystrokes(b_matched)
    # Both lists hold the identical sequence of (code, type), so pairing is
    # deterministic and the two results correspond one to one.
    assert [p["code"] for p in kp] == [p["code"] for p in bp], "pairing diverged"

    dwell_err = [b["dwell"] - k["dwell"] for k, b in zip(kp, bp)]
    rule("3. Dwell time error (how long a key is held)")
    print(describe(dwell_err))
    if kp:
        print(f"\n  for scale, kernel mean dwell = {st.mean([p['dwell'] for p in kp]):.1f} ms"
              f"   (over {len(kp)} fully matched keystrokes)")

    kf, bf = flight_times(kp), flight_times(bp)
    flight_err = [b - k for k, b in zip(kf, bf)]
    rule("4. Flight time error (gap between keys)")
    print(describe(flight_err))
    if kf:
        print(f"\n  for scale, kernel mean flight = {st.mean(kf):.1f} ms"
              f"   ({sum(1 for f in kf if f < 0)} rollovers)")

    # ------------------------------------------------------------ 5. clock ticks
    rule("5. Clock resolution")
    kr, br = resolution_ms(kernel), resolution_ms(browser)
    print(f"  kernel  : {kr if kr else '?'} ms")
    print(f"  browser : {br if br else '?'} ms")

    # ----------------------------------------------------------------- verdict
    rule("Verdict")
    k_idx = {i for i, _ in matches}
    lo, hi = min(k_idx), max(k_idx)
    lost = sum(1 for i in range(lo, hi + 1) if i not in k_idx)
    if lost:
        print(f"  ! The browser missed {lost} event(s) during the overlap. Check for a")
        print(f"    second keyboard device, or keys pressed while the page lacked focus.")
    else:
        print("  + No events lost during the overlap.")

    if dwell_err:
        p95 = p95_abs(dwell_err)
        mean_dwell = st.mean([p["dwell"] for p in kp])
        pct = 100 * p95 / mean_dwell if mean_dwell else 0
        verdict = "usable" if pct < 5 else "MARGINAL" if pct < 15 else "A PROBLEM"
        print(f"  + 95% of dwell measurements are within {p95:.2f} ms of the kernel")
        print(f"    = {pct:.1f}% of a typical dwell. This is {verdict}.")
    print()


if __name__ == "__main__":
    main()
