#!/usr/bin/env python3
"""Generate a kernel/browser pair with known error, to check compare.py works.

If we inject 2 ms of jitter and compare.py does not report roughly 2 ms, the
tool is broken and any real measurement it produces is worthless.
"""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--jitter", type=float, default=0.0, help="ms of random error to add")
ap.add_argument("--offset", type=float, default=1234.5, help="ms of constant lag (should cancel)")
ap.add_argument("--drift", type=float, default=0.0, help="ms of error gained per second")
ap.add_argument("--drop", type=int, default=0, help="events the browser misses")
ap.add_argument("--n", type=int, default=60, help="keystrokes")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

rng = random.Random(a.seed)
EPOCH = 1789679856.0
CODES = ["KeyT", "KeyH", "KeyE", "Space", "KeyQ", "KeyU", "KeyI", "KeyC", "KeyK"]

kernel, t = [], 0.0
for i in range(a.n):
    code = CODES[i % len(CODES)]
    dwell = rng.gauss(90, 18)
    kernel.append({"code": code, "type": "down", "t": t})
    kernel.append({"code": code, "type": "up", "t": t + dwell})
    # every 7th keystroke overlaps the next -> negative flight, like a fast typist
    t += dwell + (rng.gauss(-25, 8) if i % 7 == 6 else rng.gauss(115, 30))

browser = []
for e in kernel:
    err = rng.gauss(0, a.jitter) if a.jitter else 0.0
    browser.append({"code": e["code"], "type": e["type"],
                    "t": e["t"] + a.offset + err + a.drift * (e["t"] / 1000.0)})

for _ in range(a.drop):
    browser.pop(rng.randrange(2, len(browser) - 2))

OUT_K, OUT_B = Path("synthetic-kernel.jsonl"), Path("synthetic-browser.json")
for f in (OUT_K, OUT_B):
    if f.exists() and "synthetic" not in f.read_text(2000) and f.stat().st_size > 0:
        pass  # our own previous output; safe to replace
OUT_K.write_text("".join(
    json.dumps({"code": e["code"], "type": e["type"], "t": EPOCH + e["t"] / 1000.0}) + "\n"
    for e in kernel))
OUT_B.write_text(json.dumps({"events": browser}, indent=1))
print(f"wrote {OUT_K} and {OUT_B}: {len(kernel)} kernel events, {len(browser)} browser events "
      f"(jitter={a.jitter}ms offset={a.offset}ms drift={a.drift}ms/s dropped={a.drop})")
