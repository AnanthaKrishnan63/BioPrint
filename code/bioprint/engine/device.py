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

Three groups, weighted differently (project lead's call, 2026-09-19). Panopticlick
fingerprints a BROWSER; the table in CLAUDE.md asks about a different COMPUTER,
and one person may well use two browsers on one machine:

  hardware  screen, cores, memory, touch points, fonts, time-zone country, OS,
            GPU vendor family. Full weight.
  browser   UA family/version/brands, client hints, languages, plugins, canvas,
            audio, the browser-formatted GPU strings. BROWSER_WEIGHT.
            An attribute the current browser cannot report ("not reported"
            where enrollment had a value) counts here whatever its own group:
            that is the browser's doing, not the machine's.
  context   the client IP network. CONTEXT_WEIGHT. Location, not device: the
            owner's laptop moves between networks, and every device on one
            wifi shares an address. (Inert behind a USB tunnel: both ends are
            localhost.)

score = hardware bits + BROWSER_WEIGHT x browser bits + CONTEXT_WEIGHT x context bits.
A large browser share on a near-zero hardware share is reported as "same
machine, different browser: Chrome -> Firefox".
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

from contracts import Contribution, SignalResult

# Bits of mismatch above which we say "different device". A single browser
# update (ua_version 0.5 + canvas 5.0 = 5.5) or a single new monitor (screen
# 4.83 + avail 0.5 = 5.33) stays under it; two independent changes cross it, and
# a different machine scores 20+.
THRESHOLD_BITS = 6.0
# A different browser on the same machine is not a different device. Switching
# browser family moves essentially the whole browser group (~29 bits); at 0.2
# that is 5.8, under the limit on its own, while any hardware change on top of
# it (a new monitor is 5.3) crosses it.
BROWSER_WEIGHT = 0.2
CONTEXT_WEIGHT = 0.5   # a different network is where the device is, not what it is

Group = Literal["hardware", "browser", "context"]

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
    group: Group = "browser"


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


ZONEINFO = Path(os.environ.get("BIOPRINT_ZONEINFO", "/usr/share/zoneinfo"))
_ZONE_NAME = re.compile(r"^[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+){0,2}$")


@lru_cache(maxsize=1)
def _zone_countries() -> dict[str, str]:
    """IANA zone -> ISO country code(s), from tzdata's zone1970.tab (zone.tab as
    a fallback). Empty if the system has no tzdata: then zones compare as strings."""
    out: dict[str, str] = {}
    for tab in ("zone1970.tab", "zone.tab"):
        try:
            for line in (ZONEINFO / tab).read_text().splitlines():
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    out.setdefault(parts[2], parts[0])
        except OSError:
            continue
    return out


def tz_country(zone: Any) -> Any:
    """'Asia/Calcutta' -> 'IN'. Aliases resolve through the zoneinfo symlink
    ('Asia/Calcutta' is a link to 'Asia/Kolkata'). Unknown zones stay as given.
    Country, not offset: many zones share an offset, and the lead's call was
    that a same-country move is not evidence of a different machine."""
    if not isinstance(zone, str) or not _ZONE_NAME.match(zone):
        return zone
    table = _zone_countries()
    if zone in table:
        return table[zone]
    try:
        real = (ZONEINFO / zone).resolve()
        canonical = str(real.relative_to(ZONEINFO.resolve()))
    except (OSError, ValueError):
        return zone
    return table.get(canonical, zone)


def _tz_country(env: dict) -> Any:
    return tz_country(env["timezone"]) if "timezone" in env else _ABSENT


GPU_FAMILY = re.compile(r"nvidia|geforce|quadro|radeon|amd|intel|adreno|mali|apple|powervr|swiftshader|llvmpipe",
                        re.I)
_GPU_CANON = {"geforce": "nvidia", "quadro": "nvidia", "radeon": "amd"}


def gpu_family(renderer: Any, vendor: Any) -> Any:
    """'ANGLE (Intel, Mesa Intel(R) Iris(R) Xe...)' and Firefox's sanitised
    'Intel(R) HD Graphics, or similar' both -> 'intel'. The renderer string is
    browser-formatted; the vendor family is the hardware."""
    for text in (renderer, vendor):
        if isinstance(text, str) and (m := GPU_FAMILY.search(text)):
            fam = m.group(0).lower()
            return _GPU_CANON.get(fam, fam)
    return None if renderer is None and vendor is None else "other"


def _gpu_family(env: dict) -> Any:
    if "webgl_renderer" not in env and "webgl_vendor" not in env:
        return _ABSENT
    return gpu_family(env.get("webgl_renderer"), env.get("webgl_vendor"))


def ip_network(ip: Any) -> Any:
    """The network an address belongs to: /24 for IPv4, /64 for IPv6. Two
    devices on one wifi share it; the same laptop on another wifi does not."""
    if not isinstance(ip, str):
        return ip
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return ip  # e.g. "testclient": compared as a string
    bits = 24 if addr.version == 4 else 64
    return str(ipaddress.ip_network(f"{addr}/{bits}", strict=False))


def _ip_network(env: dict) -> Any:
    return ip_network(env["ip"]) if "ip" in env else _ABSENT


BROWSER_NAMES = (("Edg/", "Edge"), ("OPR/", "Opera"), ("SamsungBrowser", "Samsung Internet"),
                 ("Firefox/", "Firefox"), ("FxiOS", "Firefox"), ("CriOS", "Chrome"), ("Chrome/", "Chrome"),
                 ("Chromium/", "Chromium"), ("Safari/", "Safari"))


def browser_name(ua: Any) -> str:
    if not isinstance(ua, str):
        return "unknown browser"
    for needle, name in BROWSER_NAMES:
        if needle in ua:
            return name
    return "unknown browser"


PANOPTICLICK = "Eckersley 2010, Panopticlick"
AMIUNIQUE = "Laperdrix et al. 2016, AmIUnique (normalised entropy x 16.86)"

ATTRIBUTES: tuple[Attr, ...] = (
    # ---------------------------------------------------------------- hardware / OS
    Attr("fonts", "installed fonts", 13.9, PANOPTICLICK, _sorted_list("fonts"), "hardware"),
    Attr("screen", "screen size and colour depth", 4.83, PANOPTICLICK + " ('video')", _screen, "hardware"),
    # Panopticlick's 3.04 bits were for the zone name. We compare the COUNTRY the
    # zone belongs to: aliases (Asia/Calcutta = Asia/Kolkata) collapse, and a
    # move within a country is not evidence of another machine.
    Attr("timezone", "time zone country", 3.04, PANOPTICLICK + " (zone -> country)", _tz_country, "hardware"),
    Attr("platform", "operating system", 2.3, AMIUNIQUE + " 0.137", _plain("platform"), "hardware"),
    # research/big_idea/07 gives 2-5 bits for the hardware-count group
    Attr("hardware_concurrency", "CPU cores", 2.0, "estimate (catalogue: hardware counts 2-5)",
         _plain("hardware_concurrency"), "hardware"),
    Attr("device_memory", "device memory", 1.5, "estimate (Chromium only, capped at 8 GB)",
         _plain("device_memory"), "hardware"),
    Attr("max_touch_points", "touch points", 1.0, "estimate", _plain("max_touch_points"), "hardware"),
    Attr("avail_screen", "usable screen (taskbar/dock)", 0.5, "estimate", _avail, "hardware"),
    # AmIUnique's renderer/vendor strings are browser-formatted (ANGLE vs Mesa vs
    # Firefox's sanitised "or similar"); the vendor FAMILY is what the machine has.
    Attr("gpu_family", "GPU vendor family", 2.0, "estimate (AmIUnique vendor 0.127 = 2.1)", _gpu_family, "hardware"),
    # ---------------------------------------------------------------- browser
    # Panopticlick: 10.0 bits for the whole UA string, most of it version and
    # build numbers. Family (browser x OS x architecture) is a few dozen common
    # combinations: ~4 bits. Version is expected to change monthly.
    Attr("ua_family", "browser and OS family", 4.0, "estimate: Panopticlick 10.0 with versions stripped",
         _ua_family, "browser"),
    Attr("ua_version", "browser version", 0.5, "estimate: expected monthly drift", _ua_version, "browser"),
    Attr("ua_brands", "browser brand", 1.0, "estimate (GREASE brands ignored; implied by ua_family)",
         _ua_brands, "browser"),
    Attr("uach_platform", "client-hints platform", 0.5, "estimate (implied by platform)",
         _plain("uach_platform"), "browser"),
    Attr("languages", "preferred languages", 5.9, AMIUNIQUE + " 0.351 (content language)",
         _sorted_list("languages"), "browser"),
    # Panopticlick's 15.4 bits was for the full plugin LIST. We only have the
    # count, which today is ~5 on desktop and 0 on mobile: about one bit.
    Attr("plugins", "plugin count", 1.0, "estimate (Panopticlick 15.4 was for the full list)",
         _plain("plugins"), "browser"),
    # canvas 0.491 -> 8.3 bits, but the canvas hash moves on browser and driver
    # updates, so it is down-weighted to stay under THRESHOLD together with a
    # version bump. Firefox/Safari add noise: then it never reaches a majority
    # value at enrollment and is skipped.
    Attr("canvas_hash", "canvas rendering", 5.0, AMIUNIQUE + " 0.491 = 8.3, down-weighted for update drift",
         _plain("canvas_hash"), "browser"),
    Attr("audio_hash", "audio processing", 3.0, "estimate (catalogue: 4-8; noised in Firefox/Safari)",
         _plain("audio_hash"), "browser"),
    Attr("webgl_renderer", "GPU renderer string", 3.4, AMIUNIQUE + " 0.202", _plain("webgl_renderer"), "browser"),
    Attr("webgl_vendor", "GPU vendor string", 1.0, AMIUNIQUE + " 0.127 = 2.1, down-weighted (implied by renderer)",
         _plain("webgl_vendor"), "browser"),
    Attr("cookies_enabled", "cookies enabled", 0.353, PANOPTICLICK, _plain("cookies_enabled"), "browser"),
    Attr("do_not_track", "Do Not Track", 0.94, AMIUNIQUE + " 0.056", _plain("do_not_track"), "browser"),
    # ---------------------------------------------------------------- context
    # Set by the server from the connection, not by probe.js (env["ip"]).
    Attr("ip", "network", 3.0, "estimate (client /24 or /64)", _ip_network, "context"),
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


def _describe(attr: Attr, expected: Any, now: Any, weight: float = 1.0) -> str:
    tail = f"(+{attr.bits:.1f} bits)" if weight == 1.0 else f"(+{attr.bits:.1f} bits x {weight:g})"
    if isinstance(expected, list) and isinstance(now, list):
        gained = sorted(set(now) - set(expected))
        lost = sorted(set(expected) - set(now))
        parts = []
        if gained:
            parts.append(f"+{len(gained)} ({', '.join(gained[:3])}{'...' if len(gained) > 3 else ''})")
        if lost:
            parts.append(f"-{len(lost)} ({', '.join(lost[:3])}{'...' if len(lost) > 3 else ''})")
        change = f"{len(expected)} -> {len(now)}, " + " ".join(parts) if parts else f"{len(expected)} -> {len(now)}, reordered"
        return f"{attr.label} changed: {change} {tail}"
    return f"{attr.label} changed: {_show(expected)} -> {_show(now)} {tail}"


GROUP_WEIGHT: dict[str, float] = {"hardware": 1.0, "browser": BROWSER_WEIGHT, "context": CONTEXT_WEIGHT}
SAME_MACHINE_BITS = 2.0  # hardware bits at or under this: "same machine" for the browser-change sentence


def check(env: dict, enrolled_envs: list[dict]) -> SignalResult:
    """`env` is this attempt's probe output (plus env["ip"], set by the server);
    `enrolled_envs` the probe outputs of the accepted enrollment samples (may be
    empty for old accounts)."""
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
    raw = {"hardware": 0.0, "browser": 0.0, "context": 0.0}
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
        # "not reported" where enrollment had a value is the browser's doing.
        group = "browser" if now is None and exp is not None else attr.group
        weighted = round(attr.bits * GROUP_WEIGHT[group], 3)
        raw[group] += attr.bits
        contributions.append(Contribution(feature=f"device.{attr.key}", value=attr.bits, expected=0.0,
                                          deviation=weighted))
        changed.append((weighted, _describe(attr, exp, now, GROUP_WEIGHT[group])))

    score = round(sum(c.deviation for c in contributions), 3)
    contributions.sort(key=lambda c: -c.deviation)
    reasons = [r for _, r in sorted(changed, key=lambda x: -x[0])]
    if not contributions:
        note = f" ({len(unstable)} unstable at enrollment, ignored)" if unstable else ""
        reasons = [f"same device as enrollment: {compared} attributes agree{note}"]
    else:
        parts = [f"{raw['hardware']:.1f} hardware"]
        if raw["browser"]:
            parts.append(f"{raw['browser']:.1f} browser at {BROWSER_WEIGHT:g} weight")
        if raw["context"]:
            parts.append(f"{raw['context']:.1f} network at {CONTEXT_WEIGHT:g} weight")
        head = (f"{score:.1f} bits of identity differ from enrollment (limit {THRESHOLD_BITS:.0f}): "
                + ("more than a browser update explains" if score > THRESHOLD_BITS
                   else "consistent with an update, a resize or another browser")
                + f" [{', '.join(parts)}]")
        reasons.insert(0, head)
        if raw["browser"] and raw["hardware"] <= SAME_MACHINE_BITS:
            was = browser_name(expected.get("ua_family", (None,))[0] if "ua_family" in expected else
                               next((e.get("ua") for e in enrolled if e.get("ua")), None))
            now_name = browser_name(env.get("ua"))
            which = f"{was} -> {now_name}" if was != now_name else now_name
            reasons.insert(1, f"same machine, different browser: {which}")
        if raw["context"] and raw["hardware"] <= SAME_MACHINE_BITS and not raw["browser"]:
            reasons.insert(1, "same machine on a different network")
    return SignalResult(name="device", available=True, score=score, threshold=THRESHOLD_BITS,
                        flagged=score > THRESHOLD_BITS, contributions=contributions, reasons=reasons)
