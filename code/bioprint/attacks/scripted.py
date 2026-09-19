"""Scripted login: fabricate a sample and post it, with no browser involved.

Three attackers, in order of effort:

  --mode fixed      every hold 5 ms, every gap exactly 100 ms, headless-Chrome
                    environment. What `page.fill()` + a naive script produces.
  --mode uniform    delays drawn uniformly from 40-200 ms, holds 1-8 ms. Random,
                    but not human: the holds are impossible and the spread is wrong.
  --mode humanlike  gaussian holds (~95 +/- 30 ms) and gaps (~170 +/- 55 ms), rollover
                    on some pairs, trusted=true, a plausible desktop environment.
                    This is the honest hard case: it is built to pass engine/bot.py,
                    and it does. What stops it is the keystroke model, because the
                    attacker is guessing population-average timing, not the owner's.

    python attacks/scripted.py --url http://localhost:8001 --user alice \\
        --password ".tie5Roanl" --mode humanlike
"""

from __future__ import annotations

import argparse
import random

from common import codes_for, post_login, report

MODIFIERS = {"ShiftLeft", "ShiftRight"}

HEADLESS_ENV = {
    "probe_version": 1, "webdriver": True,
    "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "HeadlessChrome/140.0.0.0 Safari/537.36",
    "ua_brands": ["HeadlessChrome", "Chromium", "Not=A?Brand"], "ua_mobile": False,
    "plugins": 0, "mime_types": 0, "languages": ["en-US"],
    "outer_width": 0, "outer_height": 0, "inner_width": 800, "inner_height": 600,
    "screen_width": 800, "screen_height": 600, "hardware_concurrency": 1, "device_memory": 8,
    "max_touch_points": 0, "has_window_chrome": False,
    "notification_permission": "denied", "permissions_notifications": "prompt",
    "timezone": "UTC", "automation_globals": ["cdc_adoQpoasnfa76pfcZLmcfl_Array"],
    "webgl_vendor": "Google Inc.", "webgl_renderer": "Google SwiftShader",
}

HUMAN_ENV = {
    "probe_version": 1, "webdriver": False,
    "ua": "Mozilla/5.0 (X11; Linux x86_64; rv:155.0) Gecko/20100101 Firefox/155.0",
    "ua_brands": None, "ua_mobile": None, "platform": "Linux x86_64",
    "plugins": 5, "mime_types": 2, "languages": ["en-GB", "en"],
    "outer_width": 1920, "outer_height": 1050, "inner_width": 1920, "inner_height": 950,
    "screen_width": 1920, "screen_height": 1080, "hardware_concurrency": 8, "device_memory": 8,
    "max_touch_points": 0, "has_window_chrome": False, "notification_permission": "default",
    "permissions_notifications": "prompt", "timezone": "Asia/Kolkata", "automation_globals": [],
    "webgl_vendor": "AMD", "webgl_renderer": "AMD Radeon Graphics (radeonsi)",
}


def build(password: str, mode: str, rng: random.Random, trusted: bool) -> dict:
    codes = codes_for(password)
    events: list[dict] = []
    t = 500.0
    for i, code in enumerate(codes):
        if mode == "fixed":
            hold, gap = 5.0, 100.0
        elif mode == "uniform":
            hold, gap = rng.uniform(1.0, 8.0), rng.uniform(40.0, 200.0)
        else:  # humanlike
            hold = max(35.0, rng.gauss(95.0, 30.0))
            gap = max(55.0, rng.gauss(170.0, 55.0))
            if code in MODIFIERS:      # Shift is held across the next key: real rollover
                hold += gap + rng.uniform(20.0, 60.0)
                gap = rng.uniform(60.0, 120.0)
        events.append({"code": code, "type": "down", "t": round(t, 3), "field": "password", "trusted": trusted})
        events.append({"code": code, "type": "up", "t": round(t + hold, 3), "field": "password", "trusted": trusted})
        t += gap
    events.append({"code": "Enter", "type": "down", "t": round(t, 3), "field": "password", "trusted": trusted})
    events.append({"code": "Enter", "type": "up", "t": round(t + 60.0, 3), "field": "password", "trusted": trusted})
    events.sort(key=lambda e: e["t"])
    return {
        "keystrokes": events,
        "pointer": [],
        "env": dict(HUMAN_ENV if mode == "humanlike" else HEADLESS_ENV),
        "meta": {"had_paste": False, "viewport": "1920x950@1", "targets": {}, "submit_via": "enter"},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--db", default=None, help="unused; accepted so every attack shares one command line")
    p.add_argument("--mode", choices=["fixed", "uniform", "humanlike"], default="fixed")
    p.add_argument("--untrusted", action="store_true",
                   help="mark events isTrusted=false, as a browser would for dispatched events")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    sample = build(a.password, a.mode, random.Random(a.seed), trusted=not a.untrusted)
    print(f"posting a {a.mode} sample: {len(sample['keystrokes'])} key events, trusted={not a.untrusted}")
    return report(f"scripted:{a.mode}", post_login(a.url, a.user, a.password, sample))


if __name__ == "__main__":
    raise SystemExit(main())
