"""Shared helpers for the attack scripts. Standard library + httpx only.

Attack scripts live here so the demo has real villains to run against a real
server. They are also the fixtures for engine/bot.py: if a rule cannot catch its
own attack script, the rule is not finished.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

# event.code for the characters a hackathon password uses. Upper case and the
# shifted symbols need a Shift press too, exactly as a human's keyboard would send.
_PUNCT = {".": "Period", ",": "Comma", "/": "Slash", ";": "Semicolon", "'": "Quote",
          "[": "BracketLeft", "]": "BracketRight", "-": "Minus", "=": "Equal",
          "\\": "Backslash", "`": "Backquote", " ": "Space"}
_SHIFTED = {"!": "Digit1", "@": "Digit2", "#": "Digit3", "$": "Digit4", "%": "Digit5",
            "^": "Digit6", "&": "Digit7", "*": "Digit8", "(": "Digit9", ")": "Digit0",
            "_": "Minus", "+": "Equal", ":": "Semicolon", '"': "Quote", "<": "Comma",
            ">": "Period", "?": "Slash", "{": "BracketLeft", "}": "BracketRight",
            "|": "Backslash", "~": "Backquote"}


def codes_for(password: str) -> list[str]:
    """The physical keys a human types for this password, Shift presses included."""
    out: list[str] = []
    for ch in password:
        if ch.isupper():
            out += ["ShiftLeft", f"Key{ch}"]
        elif ch.islower():
            out.append(f"Key{ch.upper()}")
        elif ch.isdigit():
            out.append(f"Digit{ch}")
        elif ch in _SHIFTED:
            out += ["ShiftLeft", _SHIFTED[ch]]
        elif ch in _PUNCT:
            out.append(_PUNCT[ch])
        else:
            out.append("Unidentified")
    return out


def post_login(url: str, username: str, password: str, sample: dict[str, Any]) -> dict:
    r = httpx.post(f"{url.rstrip('/')}/api/login",
                   json={"username": username, "password": password, "sample": sample}, timeout=20.0)
    r.raise_for_status()
    return r.json()


def report(name: str, out: dict) -> int:
    """Print the verdict; exit code 0 when the attack was blocked (what we want)."""
    bot = next((s for s in out.get("signals", []) if s["name"] == "bot"), None)
    print(f"[{name}] decision = {out['decision']}")
    for why in out.get("reasons", []):
        print(f"    reason: {why}")
    if bot:
        print(f"    bot signal: score {bot['score']:.2f} / threshold {bot['threshold']:.2f}, flagged={bot['flagged']}")
        for why in bot.get("reasons", []):
            print(f"      - {why}")
    elif out["decision"] not in ("block",):
        print("    (no bot signal in the response)")
    print(json.dumps({"decision": out["decision"], "attempt_id": out.get("attempt_id")}))
    return 0 if out["decision"] == "block" else 1
