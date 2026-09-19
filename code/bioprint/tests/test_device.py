"""engine/device.py: evidence-weighted "same device?" built from per-attribute entropy.

The bar the signal has to clear: an impostor on the OWNER'S laptop must score 0
bits (the demo depends on it), a browser update must stay under the limit, and
a different laptop must be far over it. Weights are the published Panopticlick /
AmIUnique entropies, so the tests pin the numbers, not just the flags.
"""

from __future__ import annotations

import json

import pytest

from engine import device

CHROME_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.6613.84 Safari/537.36"
FONTS = ["Arial", "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif", "Liberation Mono", "Liberation Sans",
         "Liberation Serif", "Noto Sans", "Ubuntu", "Ubuntu Mono"]


def env(**over) -> dict:
    """A probe_version-2 env for an Ubuntu laptop on Chrome 128."""
    e = {
        "probe_version": 2, "webdriver": False, "ua": CHROME_UA,
        "ua_brands": ["Chromium", "Google Chrome", "Not;A=Brand"], "ua_mobile": False,
        "uach_platform": "Linux", "platform": "Linux x86_64", "plugins": 5, "mime_types": 2,
        "languages": ["en-GB", "en"], "outer_width": 1920, "outer_height": 1050,
        "inner_width": 1920, "inner_height": 950, "screen_width": 1920, "screen_height": 1080,
        "avail_width": 1920, "avail_height": 1053, "color_depth": 24, "hardware_concurrency": 8,
        "device_memory": 8, "max_touch_points": 0, "has_window_chrome": True,
        "notification_permission": "default", "permissions_notifications": "prompt",
        "timezone": "Asia/Kolkata", "automation_globals": [],
        "webgl_vendor": "Intel", "webgl_renderer": "ANGLE (Intel, Mesa Intel(R) Iris(R) Xe Graphics, OpenGL 4.6)",
        "fonts": list(FONTS), "cookies_enabled": True, "do_not_track": None,
        "canvas_hash": "9f1c2b77", "audio_hash": "124.04347", "probe_ms": 21.3,
    }
    e.update(over)
    return e


def enrolled(n: int = 10, **over) -> list[dict]:
    return [env(**over) for _ in range(n)]


def bits(attr: str) -> float:
    return next(a.bits for a in device.ATTRIBUTES if a.key == attr)


W = device.BROWSER_WEIGHT  # browser-group attributes count at this fraction of their bits


def changed(result) -> dict[str, float]:
    return {c.feature.removeprefix("device."): c.deviation for c in result.contributions}


# ---------------------------------------------------------------- same device


def test_identical_env_scores_zero_bits_and_is_not_flagged():
    r = device.check(env(), enrolled())
    assert r.name == "device" and r.available
    assert r.score == 0.0 and not r.flagged and r.contributions == []
    assert r.threshold == device.THRESHOLD_BITS
    assert "same device" in r.reasons[0]


def test_window_resize_is_free():
    """inner/outer sizes are never compared: a resized window is the same laptop."""
    r = device.check(env(inner_width=800, inner_height=600, outer_width=820, outer_height=700), enrolled())
    assert r.score == 0.0 and not r.flagged


def test_probe_ms_and_bot_only_keys_are_ignored():
    r = device.check(env(probe_ms=99.9, webdriver=True, automation_globals=["cdc_x"]), enrolled())
    assert r.score == 0.0


# ---------------------------------------------------------------- different device


def test_gpu_fonts_timezone_changed_flags_with_the_published_bits():
    other = env(webgl_renderer="ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))",
                fonts=["Arial", "Calibri", "Segoe UI", "Tahoma", "Verdana"], timezone="Europe/London")
    r = device.check(other, enrolled())
    assert r.flagged
    # hardware at full weight; the browser-formatted renderer string at a quarter
    assert changed(r) == {"fonts": 13.9, "gpu_family": 2.0, "timezone": 3.04, "webgl_renderer": pytest.approx(3.4 * W)}
    assert r.score == pytest.approx(13.9 + 2.0 + 3.04 + 3.4 * W)
    # contributions are ordered by weight and the reasons say what moved, in English
    assert [c.feature for c in r.contributions] == ["device.fonts", "device.timezone", "device.gpu_family",
                                                     "device.webgl_renderer"]
    assert any("GPU renderer string changed" in s and "SwiftShader" in s and f"+3.4 bits x {W:g}" in s for s in r.reasons)
    assert any("GPU vendor family changed: intel -> swiftshader" in s for s in r.reasons)
    assert any("time zone country changed: IN -> GB" in s for s in r.reasons)
    assert any("installed fonts changed" in s and "+4" in s and "-9" in s for s in r.reasons)
    assert "more than a browser update explains" in r.reasons[0]


def test_a_whole_different_laptop_is_far_over_the_limit():
    mac = env(ua="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                 "Version/17.5 Safari/605.1.15",
              ua_brands=None, uach_platform=None, platform="MacIntel", plugins=0, device_memory=None,
              screen_width=1512, screen_height=982, avail_width=1512, avail_height=945, color_depth=30,
              hardware_concurrency=10, webgl_vendor="Apple", webgl_renderer="Apple M2",
              fonts=["Arial", "Helvetica", "Helvetica Neue", "Menlo", "Monaco"], canvas_hash="0badf00d",
              audio_hash="35.74996")
    r = device.check(mac, enrolled())
    assert r.flagged and r.score > 25
    assert sum(c.deviation for c in r.contributions) == pytest.approx(r.score)


def test_one_new_monitor_alone_does_not_flag():
    """Docking to an external display: screen (4.83) + usable screen (0.5) < 6."""
    r = device.check(env(screen_width=2560, screen_height=1440, avail_width=2560, avail_height=1413), enrolled())
    assert changed(r) == {"screen": 4.83, "avail_screen": 0.5}
    assert not r.flagged


# ---------------------------------------------------------------- drift: browser updates


def test_version_only_ua_change_is_small():
    r = device.check(env(ua=CHROME_UA.replace("128.0.6613.84", "129.0.6668.58")), enrolled())
    assert changed(r) == {"ua_version": pytest.approx(0.5 * W)}
    assert not r.flagged
    assert "consistent with an update" in r.reasons[0]


def test_grease_brand_churn_is_free():
    """Chromium rotates the GREASE brand every release; only real brand names count."""
    r = device.check(env(ua_brands=["Chromium", "Google Chrome", "Not/A)Brand"]), enrolled())
    assert r.score == 0.0


def test_browser_family_change_costs_more_than_a_version():
    firefox = "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"
    r = device.check(env(ua=firefox, ua_brands=None, has_window_chrome=False), enrolled())
    assert changed(r)["ua_family"] == pytest.approx(4.0 * W) and changed(r)["ua_brands"] == pytest.approx(1.0 * W)
    assert "ua_version" in changed(r)
    assert not r.flagged
    assert "same machine, different browser: Chrome -> Firefox" in r.reasons[1]


def test_update_with_new_canvas_stays_under_the_limit():
    """The worst honest update: version bump plus a re-rendered canvas = 5.5 < 6."""
    r = device.check(env(ua=CHROME_UA.replace("128.0.6613.84", "129.0.6668.58"), canvas_hash="1234abcd"), enrolled())
    assert r.score == pytest.approx(5.5 * W) and not r.flagged


# ---------------------------------------------------------------- majority logic


def test_majority_value_is_the_expectation_not_the_last_sample():
    """Two of ten enrollment reps were done on the external monitor; the laptop
    screen is still the expectation, and matching it costs nothing."""
    reps = enrolled(8) + enrolled(2, screen_width=2560, screen_height=1440)
    assert device.check(env(), reps).score == 0.0
    r = device.check(env(screen_width=2560, screen_height=1440), reps)
    assert changed(r) == {"screen": 4.83}


def test_attribute_with_no_majority_is_skipped_not_counted():
    """Safari/Brave-style canvas noise: a different hash on every rep never reaches a
    majority, so the attribute is ignored instead of always costing 5 bits."""
    reps = [env(canvas_hash=f"{i:08x}") for i in range(10)]
    r = device.check(env(canvas_hash="ffffffff"), reps)
    assert "canvas_hash" not in changed(r) and r.score == 0.0
    assert "1 unstable at enrollment" in r.reasons[0]


def test_exact_half_is_not_a_majority():
    reps = enrolled(5, timezone="Asia/Kolkata") + enrolled(5, timezone="Europe/London")
    assert device.check(env(timezone="America/New_York"), reps).score == 0.0


def test_expected_values_reports_votes():
    reps = enrolled(7) + enrolled(3, hardware_concurrency=4)
    exp = device.expected_values(reps)
    assert exp["hardware_concurrency"] == (8, 7, 10)
    assert exp["fonts"][0] == sorted(FONTS)


# ---------------------------------------------------------------- old probes, missing data


def test_v1_enrollment_does_not_penalise_v2_attributes():
    """Accounts enrolled before probe_version 2 have no fonts/canvas: those keys are
    absent (not null), so they are neither expected nor counted."""
    v1 = [{k: v for k, v in env(probe_version=1).items()
           if k not in ("fonts", "cookies_enabled", "do_not_track", "canvas_hash", "audio_hash")} for _ in range(10)]
    r = device.check(env(), v1)
    assert r.available and r.score == 0.0
    r = device.check(env(timezone="Europe/London"), v1)
    assert changed(r) == {"timezone": 3.04}


def test_v2_enrollment_and_v1_attempt_compares_only_shared_attributes():
    old = {k: v for k, v in env(probe_version=1).items() if k != "fonts"}
    assert device.check(old, enrolled()).score == 0.0


def test_null_is_a_value_absent_is_not():
    """Chrome reports deviceMemory; Firefox reports null. Enrolled on one, logging in
    on the other IS a difference. A key the probe never sent is not."""
    r = device.check(env(device_memory=None), enrolled())
    assert changed(r) == {"device_memory": pytest.approx(1.5 * W)}  # counted, in the browser group
    no_key = {k: v for k, v in env().items() if k != "device_memory"}
    assert device.check(no_key, enrolled()).score == 0.0


@pytest.mark.parametrize("enrolled_envs", [[], [{}], [{"ua": CHROME_UA}], [None, 3, "x"]])
def test_unavailable_without_enrollment_probe(enrolled_envs):
    r = device.check(env(), enrolled_envs)
    assert r.name == "device" and not r.available and not r.flagged and r.score == 0.0
    assert r.reasons == ["no enrollment environment to compare against"]


@pytest.mark.parametrize("attempt", [{}, None, {"ua": CHROME_UA}, "junk"])
def test_unavailable_without_attempt_probe(attempt):
    r = device.check(attempt, enrolled())
    assert not r.available and not r.flagged
    assert r.reasons == ["browser probe did not run on this attempt"]


def test_hostile_values_do_not_crash():
    weird = env(fonts="not-a-list", languages=None, ua=12345, ua_brands="x", screen_width=None,
                canvas_hash={"a": [1, 2]}, timezone=["a"])
    r = device.check(weird, enrolled())
    assert r.available and r.score > 0
    assert device.check(env(), [weird] * 10).available


# ---------------------------------------------------------------- invariants


def test_never_a_single_hashed_id():
    """CLAUDE.md: fuzzy-match attribute vectors, never hash to one ID. Every
    contribution names one probe attribute, and its weight is that attribute's
    published entropy; nothing is a digest of the whole env."""
    r = device.check(env(timezone="UTC", platform="Win32"), enrolled())
    for c in r.contributions:
        key = c.feature.removeprefix("device.")
        assert c.feature.startswith("device.") and key in {a.key for a in device.ATTRIBUTES}
        assert c.value == bits(key) and c.expected == 0.0
        assert c.deviation == pytest.approx(bits(key) * device.GROUP_WEIGHT[
            next(a.group for a in device.ATTRIBUTES if a.key == key)])


def test_weights_add_up_to_a_full_fingerprint_and_the_threshold_is_a_small_slice():
    assert 45 < device.TOTAL_BITS < 70  # the glossary's "50+ bits"
    assert device.THRESHOLD_BITS == 6.0
    assert bits("ua_version") + bits("canvas_hash") < device.THRESHOLD_BITS  # a browser update
    assert bits("screen") + bits("avail_screen") < device.THRESHOLD_BITS  # a new monitor
    assert bits("fonts") > device.THRESHOLD_BITS  # a different OS install alone is enough


def test_result_is_json_serialisable_for_the_dashboard():
    r = device.check(env(fonts=FONTS[:3], timezone="UTC"), enrolled())
    json.dumps(r.model_dump())


# ---------------------------------------------------------------- groups and weights (lead's call, 2026-09-19)


FIREFOX_SAME_LAPTOP = dict(
    ua="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:155.0) Gecko/20100101 Firefox/155.0",
    ua_brands=None, uach_platform=None, device_memory=None, languages=["en-GB"], has_window_chrome=False,
    webgl_vendor="Intel", webgl_renderer="Intel(R) HD Graphics, or similar", timezone="Asia/Calcutta",
    canvas_hash="77b2c1f9", audio_hash="35.74997")


def test_another_browser_on_the_same_laptop_is_the_same_device():
    """The real case that motivated the split: Chrome enrollment, Firefox login,
    one laptop. Panopticlick counted it as 20.8 bits; the hardware group is 0."""
    r = device.check(env(**FIREFOX_SAME_LAPTOP), enrolled())
    assert not r.flagged
    assert r.score < device.THRESHOLD_BITS
    assert "same machine, different browser: Chrome -> Firefox" in r.reasons[1]
    assert "0.0 hardware" in r.reasons[0]
    # the only hardware-group key that moved is one Firefox does not report, and
    # it was charged at browser weight
    assert changed(r)["device_memory"] == pytest.approx(1.5 * W)


def test_time_zone_alias_is_the_same_country():
    assert device.tz_country("Asia/Calcutta") == device.tz_country("Asia/Kolkata") == "IN"
    assert device.check(env(timezone="Asia/Calcutta"), enrolled()).score == 0.0


def test_same_offset_different_country_still_counts():
    """Kolkata and Colombo share +05:30; an offset comparison would miss the move."""
    assert changed(device.check(env(timezone="Asia/Colombo"), enrolled())) == {"timezone": 3.04}


def test_unknown_zone_compares_as_a_string():
    assert device.tz_country("Mars/Olympus") == "Mars/Olympus"
    assert device.tz_country("../../etc/passwd") == "../../etc/passwd"
    assert changed(device.check(env(timezone="Mars/Olympus"), enrolled())) == {"timezone": 3.04}


def test_not_reported_is_the_browsers_doing_not_the_machines():
    """Chromium-only APIs missing in Firefox: counted, but in the browser group."""
    r = device.check(env(device_memory=None, ua_brands=None, uach_platform=None), enrolled())
    assert changed(r) == {"device_memory": pytest.approx(1.5 * W), "ua_brands": pytest.approx(1.0 * W),
                          "uach_platform": pytest.approx(0.5 * W)}
    assert not r.flagged


def test_gpu_family_survives_firefox_sanitising_and_catches_a_real_change():
    assert device.gpu_family("Intel(R) HD Graphics, or similar", None) == "intel"
    assert device.gpu_family("ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0)", "Google Inc.") == "nvidia"
    sanitised = device.check(env(webgl_renderer="Intel(R) HD Graphics, or similar"), enrolled())
    assert changed(sanitised) == {"webgl_renderer": pytest.approx(3.4 * W)}  # string moved, hardware did not
    phone = device.check(env(webgl_renderer="Adreno (TM) 740", webgl_vendor="Qualcomm"), enrolled())
    assert changed(phone) == {"gpu_family": 2.0, "webgl_renderer": pytest.approx(3.4 * W),
                              "webgl_vendor": pytest.approx(1.0 * W)}


def test_network_is_context_at_half_weight():
    home = enrolled(ip="10.42.0.17")
    assert device.check(env(ip="10.42.0.99"), home).score == 0.0  # same /24
    r = device.check(env(ip="10.43.7.1"), home)
    assert changed(r) == {"ip": 1.5} and not r.flagged
    assert "same machine on a different network" in r.reasons[1]
    assert device.check(env(ip="10.43.7.1"), enrolled()).score == 0.0  # no address at enrollment: no evidence
    assert device.ip_network("2001:db8:1:2:3:4:5:6") == "2001:db8:1:2::/64"


def test_a_phone_is_still_far_over_the_limit_on_hardware_alone():
    phone = env(ua="Mozilla/5.0 (Android 17; Mobile; rv:156.0) Gecko/156.0 Firefox/156.0", ua_brands=None,
                uach_platform=None, device_memory=None, platform="Linux armv81", screen_width=414,
                screen_height=920, avail_width=414, avail_height=920, hardware_concurrency=8,
                max_touch_points=5, webgl_vendor="Qualcomm", webgl_renderer="Adreno (TM) 740",
                fonts=["Roboto"], plugins=0)
    r = device.check(phone, enrolled())
    hardware = sum(c.deviation for c in r.contributions
                   if c.feature.removeprefix("device.") in {a.key for a in device.ATTRIBUTES if a.group == "hardware"})
    assert r.flagged and hardware > device.THRESHOLD_BITS
    assert not any("same machine" in s for s in r.reasons)


def test_group_weights_are_what_was_agreed():
    assert device.GROUP_WEIGHT == {"hardware": 1.0, "browser": W, "context": 0.5}
    assert 0 < W < 0.5 < 1.0
    # switching browser family alone can never cross the limit...
    browser_total = sum(a.bits for a in device.ATTRIBUTES if a.group == "browser")
    assert browser_total * W < device.THRESHOLD_BITS
    # ... but with one new monitor on top it does
    assert browser_total * W + bits("screen") + bits("avail_screen") > device.THRESHOLD_BITS
    assert {a.group for a in device.ATTRIBUTES} == {"hardware", "browser", "context"}
