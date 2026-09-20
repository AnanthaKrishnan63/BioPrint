"""Non-human / replay detection. OWNER: Agent B (bot).

A separate signal, never folded into the behaviour score: a bot is not "an
unusual owner", it is a different kind of fraud and gets its own reasons.

Scoring. Every rule that fires adds its weight to `score` and one human-readable
reason. STRONG rules weigh 1.0 and block on their own; WEAK tells weigh less and
must stack (two typical weak tells exceed THRESHOLD). flagged = score > THRESHOLD.
Each fired rule is also a Contribution named "bot.<rule>", so the dashboard can
show exactly which rule fired.

False human flags are possible: these rules are heuristics, not physical proofs.
The strict public-data audit is in scripts/bot_rule_validation.py. In particular,
  - rollover makes UD negative and DD tiny, so UD is never tested and
    DD floors need MANY pairs;
  - nothing about the pointer is required when the form was submitted with Enter;
  - software rendering (llvmpipe on Linux/VMs), no plugins, etc. are weak only.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from contracts import Contribution, FeatureVector, Sample, SignalResult
from engine.features import password_keystrokes, is_virtual_keyboard

THRESHOLD = 0.75  # one strong rule (1.0) or two weak tells (>= 0.4 each)
STRONG = 1.0

# --- key-count rule
MODIFIERS = {"ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight",
             "MetaLeft", "MetaRight", "OSLeft", "OSRight", "CapsLock", "Fn", "AltGraph"}
NON_CHAR = MODIFIERS | {"Enter", "NumpadEnter", "Tab", "Backspace", "Delete", "Escape",
                        "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"}

# --- timing rules (milliseconds)
# A short-hold heuristic, not a universal hardware or human lower bound.
# Genuine public CMU dev recordings can fall below it. Browser automation can
# also produce short down/up intervals; timing alone cannot establish its origin.
HOLD_FLOOR_MS = 10.0
# Two keys in the same USB report can share a timestamp (DD = 0) during rollover, so a
# tiny DD is only suspicious when it is the norm, not once.
DD_FLOOR_MS = 10.0
DD_FLOOR_FRACTION = 0.5
# Human coefficient of variation of DD / hold within one password is ~0.2-0.6. A fixed
# delay bot is ~0; a real browser adds 1-4 ms of event-loop jitter, i.e. CV ~0.01-0.03.
MIN_CV = 0.03
MIN_TIMING_N = 5  # CV over fewer intervals is too noisy to judge a human by
IDENTICAL_TOL_MS = 0.5  # "all intervals identical": spread within timer resolution
ROUND_GRID_MS = 10.0  # fabricated samples use round numbers; a 1 ms clock gives 1/10 odds per value

# --- replay
# Mean absolute difference (ms) over all H/DD/UD features between this attempt and a
# stored one.
# Legacy exploratory rationale (not a current sealed-test result), measured on
# CMU DSL-StrongPasswordData (51 subjects x 400 repetitions,
# ~4M same-subject pairs): the two closest repetitions ANY subject ever produced differ
# by 4.74 ms MAE; the median subject's closest pair is 7.9 ms, and that is the closest
# of 79,800 pairs -- with ~10 stored samples per user the expected minimum is far higher.
# Replay floor: replaying a recording through a real browser or input injection adds only
# timer quantisation (1 ms) plus the 2.8 ms p95 delivery jitter measured in
# research/big_idea/12 Measurements.md. 4 ms sits just under the human floor.
# Cost of the gap: an attacker who fuzzes every timestamp by +/- 10 ms clears this rule
# (attacks/replay.py --jitter 10) and must then be caught by the keystroke model.
REPLAY_MAE_MS = 4.0

# --- pointer (pixels)
# The pointer's last known position before a click must be near the click. Pointer
# capture may throttle moves (a fast flick covers ~30 px per 16 ms frame), so a jump
# needs to be large to count as a teleport.
TELEPORT_PX = 150.0

SOFTWARE_GL = re.compile(r"swiftshader|llvmpipe|softpipe|software|mesa offscreen", re.I)
MOBILE_UA = re.compile(r"Android|iPhone|iPad|Mobile", re.I)
CHROME_UA = re.compile(r"Chrome/|Chromium/", re.I)
NOT_REALLY_CHROME = re.compile(r"CriOS|Edg/|OPR/|SamsungBrowser", re.I)  # window.chrome not guaranteed


@dataclass
class _Acc:
    rules: list[tuple[str, float, float, float, str]] = field(default_factory=list)

    def add(self, rule: str, weight: float, reason: str, value: float = 1.0, expected: float = 0.0) -> None:
        self.rules.append((rule, weight, value, expected, reason))

    def result(self) -> SignalResult:
        score = sum(w for _, w, _, _, _ in self.rules)
        ordered = sorted(self.rules, key=lambda r: -r[1])
        return SignalResult(
            name="bot",
            score=round(score, 3),
            threshold=THRESHOLD,
            flagged=score > THRESHOLD,
            contributions=[Contribution(feature=f"bot.{r}", value=v, expected=e, deviation=w)
                           for r, w, v, e, _ in ordered],
            reasons=[f"{why} [{'strong' if w >= STRONG else 'weak'}]" for _, w, _, _, why in ordered],
        )


def _cv(xs: list[float]) -> float:
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m > 0 else 0.0


# ------------------------------------------------------------------ rule groups


def _trust(sample: Sample, acc: _Acc) -> None:
    n = sum(not e.trusted for e in sample.keystrokes)
    if n:
        acc.add("untrusted_keys", STRONG, f"{n} key events were generated by a script (isTrusted=false)", n)
    n = sum(not e.trusted for e in sample.pointer)
    if n:
        acc.add("untrusted_pointer", STRONG, f"{n} pointer events were generated by a script (isTrusted=false)", n)


def _env(env: dict, acc: _Acc) -> None:
    if not env:
        acc.add("no_probe", 0.4, "browser probe did not run (request did not come from the login page)")
        return
    if env.get("webdriver") is True:
        acc.add("webdriver", STRONG, "browser reports it is automated (navigator.webdriver)")
    globs = env.get("automation_globals") or []
    if isinstance(globs, list) and globs:
        acc.add("automation_globals", STRONG,
                f"automation driver objects present in the page ({', '.join(map(str, globs[:3]))})", len(globs))
    ua = str(env.get("ua") or "")
    brands = [str(b) for b in (env.get("ua_brands") or []) if isinstance(b, str)]
    if "headless" in ua.lower() or any("headless" in b.lower() for b in brands):
        acc.add("headless_ua", STRONG, "browser identifies itself as headless (HeadlessChrome)")

    mobile = bool(MOBILE_UA.search(ua)) or env.get("ua_mobile") is True
    renderer = f"{env.get('webgl_vendor') or ''} {env.get('webgl_renderer') or ''}"
    if SOFTWARE_GL.search(renderer):
        # Weak on purpose: Linux without GPU drivers, VMs and remote desktops use llvmpipe.
        acc.add("software_gl", 0.25, f"graphics are software-rendered ({renderer.strip()[:60]}), typical of headless/VMs")
    if env.get("outer_width") == 0 or env.get("outer_height") == 0:
        acc.add("zero_outer_window", 0.5, "browser window has zero outer size (headless)")
    if not mobile and env.get("plugins") == 0 and ua:
        acc.add("no_plugins", 0.3, "desktop browser reports no plugins (real desktop Chrome/Firefox list the PDF viewer)")
    langs = env.get("languages")
    if isinstance(langs, list) and not langs:
        acc.add("no_languages", 0.3, "browser reports no languages")
    if CHROME_UA.search(ua) and not NOT_REALLY_CHROME.search(ua) and env.get("has_window_chrome") is False:
        acc.add("chrome_object_missing", 0.4, "Chrome user agent without the window.chrome object")
    if env.get("notification_permission") == "denied" and env.get("permissions_notifications") == "prompt":
        acc.add("permission_mismatch", 0.5, "notification permission contradicts itself (headless Chrome quirk)")
    if (MOBILE_UA.search(ua) or env.get("ua_mobile") is True) and env.get("max_touch_points") == 0:
        acc.add("mobile_no_touch", 0.4, "user agent claims a phone/tablet but the device has no touch points")
    hc = env.get("hardware_concurrency")
    if isinstance(hc, (int, float)) and not isinstance(hc, bool) and hc <= 1:
        acc.add("single_core", 0.15, "a single CPU core (common in containers)", hc)


def _key_count(sample: Sample, password_length: int, acc: _Acc) -> None:
    """Every character needs at least one non-modifier keydown (Shift adds events, never
    removes them; Firefox phantom keydowns only add). So fewer presses than characters
    means the value was set some other way."""
    presses = sum(1 for e in sample.keystrokes
                  if e.field == "password" and e.type == "down" and e.code not in NON_CHAR)
    if presses >= password_length:
        return
    if sample.meta.had_paste:
        acc.add("paste", 0.5, "password was pasted rather than typed", presses, password_length)
    elif presses == 0:
        acc.add("no_typing", STRONG, "password arrived with no key presses at all "
                "(set by a script, or autofilled by a password manager)", 0, password_length)
    else:
        acc.add("too_few_keys", STRONG, f"only {presses} key presses for a {password_length}-character password "
                "(value was set by a script)", presses, password_length)


def _timing(sample: Sample, acc: _Acc) -> None:
    # A touch keyboard reports 0 ms holds for every key: mechanical-looking, but
    # human. There is no rhythm to judge either way, so no timing rule applies.
    if is_virtual_keyboard(sample):
        return
    ks = [k for k in password_keystrokes(sample) if k.code not in MODIFIERS]
    if not ks:
        return
    holds = [k.up - k.down for k in ks]
    dds = [b.down - a.down for a, b in zip(ks, ks[1:])]

    short = [h for h in holds if h < HOLD_FLOOR_MS]
    if short:
        acc.add("impossible_hold", STRONG, f"{len(short)} keys held under the configured {HOLD_FLOOR_MS:.0f} ms timing floor "
                f"(min {min(short):.1f} ms)", min(short), HOLD_FLOOR_MS)

    if len(dds) >= MIN_TIMING_N - 1:
        tiny = sum(d < DD_FLOOR_MS for d in dds)
        if tiny / len(dds) > DD_FLOOR_FRACTION:
            acc.add("impossible_dd", STRONG, f"{tiny} of {len(dds)} consecutive key presses under "
                    f"{DD_FLOOR_MS:.0f} ms apart", tiny / len(dds), DD_FLOOR_FRACTION)

    for label, xs in (("DD", dds), ("hold", holds)):
        if len(xs) < 3:
            continue
        if max(xs) - min(xs) <= IDENTICAL_TOL_MS:
            acc.add(f"identical_{label.lower()}", STRONG,
                    f"every {label} interval is identical ({xs[0]:.1f} ms): mechanical input", 0.0)
            continue
        if len(xs) >= MIN_TIMING_N and (cv := _cv(xs)) < MIN_CV:
            acc.add(f"low_variance_{label.lower()}", STRONG,
                    f"{label} timing too regular for a human (CV {cv:.3f} < {MIN_CV})", cv, MIN_CV)

    intervals = holds + dds
    if len(intervals) >= 2 * MIN_TIMING_N - 1 and all(
            abs(x / ROUND_GRID_MS - round(x / ROUND_GRID_MS)) < 1e-6 for x in intervals):
        # Weak: Firefox resistFingerprinting / Tor coarsen timers to 100 ms, which would pass this.
        acc.add("round_timings", 0.5, f"every interval is a multiple of {ROUND_GRID_MS:.0f} ms (fabricated timings)")


def _replay(vector: FeatureVector | None, prior_vectors: list[list[float]], acc: _Acc) -> None:
    if vector is None or not prior_vectors:
        return
    cur = vector.values
    best = min((sum(abs(a - b) for a, b in zip(cur, p)) / len(cur)
                for p in prior_vectors if len(p) == len(cur) and cur), default=None)
    if best is not None and best < REPLAY_MAE_MS:
        acc.add("replay", STRONG, f"timing matches a stored attempt to within {best:.2f} ms on average "
                f"(humans vary by 10+ ms): replayed input", best, REPLAY_MAE_MS)


def _pointer(sample: Sample, acc: _Acc) -> None:
    if sample.meta.submit_via != "click":
        return  # Enter or unknown: pointer is not needed, so its absence says nothing
    pts = sorted(sample.pointer, key=lambda p: p.t)
    if not pts:
        acc.add("click_without_pointer", 0.4, "form was submitted by click but no pointer events were recorded")
        return
    downs = [i for i, p in enumerate(pts) if p.type == "down"]
    if not downs:
        return
    i = downs[-1]
    click = pts[i]
    if click.pointer_type != "mouse":
        return  # touch and pen have no hover: a tap legitimately appears from nowhere
    if not any(p.type == "move" for p in pts[:i]):
        # Weak: the cursor may already have rested on the button from the previous attempt.
        acc.add("click_without_movement", 0.4, "clicked without any pointer movement beforehand")
        return
    fixes = [p for p in pts[:i] if p.type in ("move", "enter", "up")]
    if fixes:
        last = fixes[-1]
        jump = ((click.x - last.x) ** 2 + (click.y - last.y) ** 2) ** 0.5
        if jump > TELEPORT_PX:
            acc.add("pointer_teleport", STRONG, f"pointer jumped {jump:.0f} px to the click with no movement "
                    "in between (synthetic click)", jump, TELEPORT_PX)


def check(
    sample: Sample,
    vector: FeatureVector | None,
    prior_vectors: list[list[float]],
    password_length: int,
) -> SignalResult:
    """`vector` is this attempt's keystroke vector (None if unscorable);
    `prior_vectors` are this user's stored vectors, for replay detection;
    `password_length` is len(password) of the (correct) password submitted, so a
    script that sets the field's value without typing is caught."""
    acc = _Acc()
    for rule in (lambda: _trust(sample, acc),
                 lambda: _env(sample.env if isinstance(sample.env, dict) else {}, acc),
                 lambda: _key_count(sample, password_length, acc),
                 lambda: _timing(sample, acc),
                 lambda: _replay(vector, prior_vectors, acc),
                 lambda: _pointer(sample, acc)):
        try:
            rule()
        except Exception as exc:  # a malformed sample must not crash login; say so instead
            acc.add("probe_error", 0.0, f"bot rule skipped: {type(exc).__name__}")
    return acc.result()
