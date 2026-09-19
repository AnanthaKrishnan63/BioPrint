"""Replay attack: steal a recorded login and post it again, byte for byte.

The strongest attack against naive keystroke biometrics. The attacker has the
password AND a recording of the owner typing it (malware, a compromised logger, a
leaked database), so the behavioural model matches perfectly. Only the replay rule
in engine/bot.py separates it from the owner: a human never reproduces their own
timing to the millisecond.

    python attacks/replay.py --url http://localhost:8001 --user alice \\
        --password ".tie5Roanl" --db /tmp/demo.db [--jitter 0] [--kind enroll]

--jitter N adds uniform +/- N ms to every event timestamp, which is what a careful
attacker would do. Sweep it (0, 2, 5, 10, 25) to find where the rule stops firing.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys

from common import post_login, report


def load_sample(db: str, user: str, kind: str | None, index: int) -> dict:
    conn = sqlite3.connect(db)
    q = ("SELECT s.sample_json FROM samples s JOIN users u ON u.id = s.user_id"
         " WHERE u.username = ?")
    args: list = [user]
    if kind:
        q += " AND s.kind = ?"
        args.append(kind)
    q += " ORDER BY s.id"
    rows = conn.execute(q, args).fetchall()
    conn.close()
    if not rows:
        sys.exit(f"no stored samples for {user!r} in {db}")
    return json.loads(rows[index % len(rows)][0])


def jitter_sample(sample: dict, ms: float, seed: int | None) -> dict:
    """Perturb every timestamp. The attacker's cheapest evasion of a replay check."""
    if ms <= 0:
        return sample
    rng = random.Random(seed)
    for stream in ("keystrokes", "pointer"):
        for e in sample.get(stream, []):
            e["t"] = e["t"] + rng.uniform(-ms, ms)
    return sample


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True, help="the stolen password (the attack assumes it is known)")
    p.add_argument("--db", required=True, help="path to the server's sqlite DB (the 'stolen recording')")
    p.add_argument("--kind", choices=["enroll", "login"], default=None, help="which stored sample to replay")
    p.add_argument("--index", type=int, default=-1, help="which stored sample, in id order (default: the newest)")
    p.add_argument("--jitter", type=float, default=0.0, help="uniform +/- ms added to every timestamp")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    sample = jitter_sample(load_sample(a.db, a.user, a.kind, a.index), a.jitter, a.seed)
    n = len(sample.get("keystrokes", []))
    print(f"replaying a stored sample: {n} key events, {len(sample.get('pointer', []))} pointer events, "
          f"jitter +/-{a.jitter:g} ms")
    return report("replay", post_login(a.url, a.user, a.password, sample))


if __name__ == "__main__":
    raise SystemExit(main())
