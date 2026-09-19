"""Device axis: is this the browser the profile was enrolled on? OWNER: device agent.

Advisory, never a block on its own: the brief's demo is an impostor on the
owner's own laptop, where this signal correctly says "same device". It exists
so the dashboard can show the disagreement pattern (behaviour off, device same
= someone at the owner's machine; both off = a different person elsewhere).

Method: Panopticlick, turned around. Eckersley (2010, "How Unique Is Your Web
Browser?") measured how many bits of identifying information each browser
attribute carries across a population. We use those same bits as EVIDENCE
WEIGHTS: an attribute that differs from enrollment contributes its published
entropy to the score, so `score` reads as "how many bits of identity moved".
Two things follow from doing it per attribute rather than hashing everything
into one ID (CLAUDE.md: never hash a fingerprint into one ID):

  - a browser update changes the version part of the UA and (sometimes) the
    canvas hash, and nothing else. Those attributes carry little weight or are
    compared on family only, so an update scores well under the threshold;
  - a different laptop changes fonts, GPU, screen, canvas at once, i.e. 20-30
    bits, and no single attribute has to be trusted for that.

Expected values come from the MAJORITY of the enrollment samples, so one
resized or noised repetition does not poison the template, and an attribute
that never settled on a majority value (Firefox/Safari canvas noise, zoom
changing screen.width in Firefox) is skipped instead of counted.

Sources for the weights, per attribute, are in ATTRIBUTES below. Where the
figure is not a published measurement it says "estimate".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from contracts import Contribution, SignalResult

# Bits of mismatch above which we say "different device". A single browser
# update (ua_version 0.5 + canvas 5.0 = 5.5) or a single new monitor (screen
# 4.83 + avail 0.5 = 5.33) stays under it; two independent changes cross it, and
# a different machine scores 20+.
THRESHOLD_BITS = 6.0

# Laperdrix et al. 2016 report NORMALISED entropy (H / log2 N) over N = 118,934
# AmIUnique fingerprints; log2(118934) = 16.86 turns that back into bits.
# The conversion is ours; the normalised values are from their Table 3.
_AMIUNIQUE_BITS = 16.86

_ABSENT = object()  # the probe did not record this attribute (old probe_version)

VERSION = re.compile(r"\d+(?:[._]\d+)*")
GREASE_BRAND = re.compile(r"[^A-Za-z0-9 ]")  # "Not/A)Brand", "Not(A:Brand": change every release


@dataclass(frozen=True)
class Attr:
    key: str  # Contribution.feature is "device.<key>"
    label: str  # plain English, for reasons
    bits: float
    source: str
    extract: Callable[[dict], Any]  # env -> JSON-able value, or _ABSENT


def _plain(*keys: str) -> Callable[[dict], Any]:
    """The attribute is one probe key, used as-is (null is a value; a missing key is absent)."""
    key = keys[0]
    return lambda env: env[key] if key in env else _ABSENT


def _ua_family(env: dict) -> Any:
    if "ua" not in env:
        return _ABSENT
    ua = env.get("ua")
    return VERSION.sub("#", ua) if isinstance(ua, str) else ua


def _ua_version(env: dict) -> Any:
    if "ua" not in env:
        return _ABSENT
    ua = env.get("ua")
    return VERSION.findall(ua) if isinstance(ua, str) else ua


def _ua_brands(env: dict) -> Any:
    if "ua_brands" not in env:
        return _ABSENT
    brands = env.get("ua_brands")
    if not isinstance(brands, list):
        return brands
    return sorted({str(b) for b in brands if not GREASE_BRAND.search(str(b))})


def _screen(env: dict) -> Any:
    keys = ("screen_width", "screen_height", "color_depth")
    if not all(k in env for k in keys):
        return _ABSENT
    return f"{env['screen_width']}x{env['screen_height']}@{env['color_depth']}"


def _avail(env: dict) -> Any:
    keys = ("avail_width", "avail_height")
    if not all(k in env for k in keys):
        return _ABSENT
    return f"{env['avail_width']}x{env['avail_height']}"


def _sorted_list(key: str) -> Callable[[dict], Any]:
    def f(env: dict) -> Any:
        if key not in env:
            return _ABSENT
        v = env.get(key)
        return sorted(str(x) for x in v) if isinstance(v, list) else v
    return f


PANOPTICLICK = "Eckersley 2010, Panopticlick"
AMIUNIQUE = "Laperdrix et al. 2016, AmIUnique (normalised entropy x 16.86)"

ATTRIBUTES: tuple[Attr, ...] = (
    # ---- measured by Panopticlick (Eckersley 2010, Table: entropy per variable)
    Attr("fonts", "installed fonts", 13.9, PANOPTICLICK, _sorted_list("fonts")),
    Attr("screen", "screen size and colour depth", 4.83, PANOPTICLICK + " ('video')", _screen),
    Attr("timezone", "time zone", 3.04, PANOPTICLICK, _plain("timezone")),
    Attr("cookies_enabled", "cookies enabled", 0.353, PANOPTICLICK, _plain("cookies_enabled")),
    # Panopticlick's 15.4 bits was for the full plugin LIST. We only have the
    # count, which today is ~5 on desktop and 0 on mobile: about one bit.
    Attr("plugins", "plugin count", 1.0, "estimate (Panopticlick 15.4 was for the full list)", _plain("plugins")),
    # ---- measured by AmIUnique (Laperdrix 2016), converted from normalised entropy
    # canvas 0.491 -> 8.3 bits, but the canvas hash moves on browser and driver
    # updates, so it is down-weighted to stay under THRESHOLD together with a
    # version bump. Firefox/Safari add noise: then it never reaches a majority
    # value at enrollment and is skipped.
    Attr("canvas_hash", "canvas rendering", 5.0, AMIUNIQUE + " 0.491 = 8.3, down-weighted for update drift", _plain("canvas_hash")),
    Attr("webgl_renderer", "GPU renderer", 3.4, AMIUNIQUE + " 0.202", _plain("webgl_renderer")),
    # vendor 0.127 -> 2.1 bits, but it is almost fully implied by the renderer.
    Attr("webgl_vendor", "GPU vendor", 1.0, AMIUNIQUE + " 0.127 = 2.1, down-weighted (implied by renderer)", _plain("webgl_vendor")),
    Attr("languages", "preferred languages", 5.9, AMIUNIQUE + " 0.351 (content language)", _sorted_list("languages")),
    Attr("platform", "operating system", 2.3, AMIUNIQUE + " 0.137", _plain("platform")),
    Attr("do_not_track", "Do Not Track", 0.94, AMIUNIQUE + " 0.056", _plain("do_not_track")),
    # ---- user agent, split so that a version bump is cheap
    # Panopticlick: 10.0 bits for the whole string, most of it version and build
    # numbers. Family (browser x OS x architecture) is a few dozen common
    # combinations: ~4 bits. Version is expected to change monthly.
    Attr("ua_family", "browser and OS family", 4.0, "estimate: Panopticlick 10.0 with versions stripped", _ua_family),
    Attr("ua_version", "browser version", 0.5, "estimate: expected monthly drift", _ua_version),
    Attr("ua_brands", "browser brand", 1.0, "estimate (GREASE brands ignored; implied by ua_family)", _ua_brands),
    Attr("uach_platform", "client-hints platform", 0.5, "estimate (implied by platform)", _plain("uach_platform")),
    # ---- hardware counts; research/big_idea/07 gives 2-5 bits for the group
    Attr("hardware_concurrency", "CPU cores", 2.0, "estimate (catalogue: hardware counts 2-5)", _plain("hardware_concurrency")),
    Attr("device_memory", "device memory", 1.5, "estimate (Chromium only, capped at 8 GB)", _plain("device_memory")),
    Attr("max_touch_points", "touch points", 1.0, "estimate", _plain("max_touch_points")),
    Attr("avail_screen", "usable screen (taskbar/dock)", 0.5, "estimate", _avail),
    Attr("audio_hash", "audio processing", 3.0, "estimate (catalogue: 4-8; noised in Firefox/Safari)", _plain("audio_hash")),
)
TOTAL_BITS = sum(a.bits for a in ATTRIBUTES)  # ~56: the "50+ bits" of the glossary


# ------------------------------------------------------------------ comparison


def _canon(value: Any) -> str:
    """Hashable, order-stable form for majority counting and equality."""
    return json.dumps(value, sort_keys=True, default=str)


def expected_values(enrolled_envs: list[dict]) -> dict[str, tuple[Any, int, int]]:
    """Per attribute: (majority value, votes, envs where it was recorded).
    Attributes with no strict majority among the envs that recorded them are left out."""
    out: dict[str, tuple[Any, int, int]] = {}
    for attr in ATTRIBUTES:
        seen: list[Any] = [attr.extract(e) for e in enrolled_envs if isinstance(e, dict)]
        seen = [v for v in seen if v is not _ABSENT]
        if not seen:
            continue
        counts = Counter(_canon(v) for v in seen)
        canon, votes = counts.most_common(1)[0]
        if votes * 2 > len(seen):
            out[attr.key] = (json.loads(canon), votes, len(seen))
    return out


def _show(v: Any) -> str:
    if v is None:
        return "not reported"
    if isinstance(v, list):
        return f"{len(v)} items"
    s = str(v)
    return s if len(s) <= 48 else s[:45] + "..."


def _describe(attr: Attr, expected: Any, now: Any) -> str:
    if isinstance(expected, list) and isinstance(now, list):
        gained = sorted(set(now) - set(expected))
        lost = sorted(set(expected) - set(now))
        parts = []
        if gained:
            parts.append(f"+{len(gained)} ({', '.join(gained[:3])}{'...' if len(gained) > 3 else ''})")
        if lost:
            parts.append(f"-{len(lost)} ({', '.join(lost[:3])}{'...' if len(lost) > 3 else ''})")
        change = f"{len(expected)} -> {len(now)}, " + " ".join(parts) if parts else f"{len(expected)} -> {len(now)}, reordered"
        return f"{attr.label} changed: {change} (+{attr.bits:.1f} bits)"
    return f"{attr.label} changed: {_show(expected)} -> {_show(now)} (+{attr.bits:.1f} bits)"


def check(env: dict, enrolled_envs: list[dict]) -> SignalResult:
    """`env` is this attempt's probe output; `enrolled_envs` the probe outputs of
    the accepted enrollment samples (may be empty for old accounts)."""
    enrolled = [e for e in (enrolled_envs or []) if isinstance(e, dict) and e.get("probe_version")]
    if not enrolled:
        return SignalResult(name="device", available=False, threshold=THRESHOLD_BITS,
                            reasons=["no enrollment environment to compare against"])
    if not isinstance(env, dict) or not env.get("probe_version"):
        return SignalResult(name="device", available=False, threshold=THRESHOLD_BITS,
                            reasons=["browser probe did not run on this attempt"])

    expected = expected_values(enrolled)
    contributions: list[Contribution] = []
    changed: list[tuple[float, str]] = []
    compared = 0
    unstable = [a.key for a in ATTRIBUTES if a.key not in expected
                and any(a.extract(e) is not _ABSENT for e in enrolled)]
    for attr in ATTRIBUTES:
        if attr.key not in expected:
            continue
        now = attr.extract(env)
        if now is _ABSENT:
            continue  # this client's probe is older than enrollment's: no evidence either way
        compared += 1
        exp = expected[attr.key][0]
        if _canon(now) == _canon(exp):
            continue
        contributions.append(Contribution(feature=f"device.{attr.key}", value=1.0, expected=0.0, deviation=attr.bits))
        changed.append((attr.bits, _describe(attr, exp, now)))

    score = round(sum(c.deviation for c in contributions), 2)
    contributions.sort(key=lambda c: -c.deviation)
    reasons = [r for _, r in sorted(changed, key=lambda x: -x[0])]
    if not contributions:
        note = f" ({len(unstable)} unstable at enrollment, ignored)" if unstable else ""
        reasons = [f"same device as enrollment: {compared} attributes agree{note}"]
    elif score > THRESHOLD_BITS:
        reasons.insert(0, f"{score:.1f} bits of identity differ from enrollment (limit {THRESHOLD_BITS:.0f}): "
                          "more than a browser update explains")
    else:
        reasons.insert(0, f"{score:.1f} bits differ (limit {THRESHOLD_BITS:.0f}): consistent with an update or a resize")
    return SignalResult(name="device", available=True, score=score, threshold=THRESHOLD_BITS,
                        flagged=score > THRESHOLD_BITS, contributions=contributions, reasons=reasons)
