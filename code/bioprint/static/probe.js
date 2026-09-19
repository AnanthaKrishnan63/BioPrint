// probe.js — browser environment probe for bot detection. OWNER: Agent B (bot).
//
// Collects cheap automation tells and sends them as Sample.env. engine/bot.py owns
// the keys and decides what they mean; this file only observes. Budget: < 50 ms,
// and it must NEVER throw — every probe is wrapped, a failing probe records null.
//
// Strength of each tell (how bot.py weighs it):
//   STRONG (can block on its own)
//     webdriver            navigator.webdriver === true. Set by WebDriver/CDP automation
//                          unless the bot patches it out.
//     automation_globals   cdc_* (chromedriver), __playwright*, __pwInitScripts, callPhantom,
//                          __nightmare, __selenium*, domAutomation... injected by drivers.
//     ua / ua_brands       "HeadlessChrome" in the UA string or UA-CH brand list.
//   WEAK (only accumulate; each is common on some real machines)
//     webgl_vendor/renderer  SwiftShader / llvmpipe / softpipe = software rendering.
//                          Typical of headless Chrome and VMs, BUT real users on Linux
//                          without GPU drivers, in VMs or over remote desktop also get
//                          llvmpipe. Software rendering alone must never block.
//     outer_width/height   0 in old headless Chrome. Real windows are never 0x0.
//     plugins/mime_types   0 on desktop in old headless. Mobile browsers legitimately 0.
//     languages            empty list in some headless builds.
//     has_window_chrome    window.chrome missing on a Chrome UA (old headless).
//     notification_permission vs permissions_notifications: "denied" vs "prompt" is the
//                          classic headless inconsistency (real Chrome agrees with itself).
//     max_touch_points     0 on a UA claiming to be mobile = UA spoofing.
//     hardware_concurrency 1 is rare on real hardware, common in containers.
//   CONTEXT ONLY (collected, no rule yet: no server-side comparison point)
//     timezone, screen size, device_memory, color depth, platform, uach_platform.

const PERMISSION_TIMEOUT_MS = 30; // permissions.query is async; never let it hold up login
const AUTOMATION_GLOBAL = /^(cdc_|\$cdc_|\$wdc_|__playwright|__pw|__puppeteer|callPhantom$|_phantom$|__nightmare$|__selenium|_selenium$|__webdriver|__driver_|__fxdriver|domAutomation|_Selenium_IDE_Recorder$)/;

function safe(fn, fallback = null) {
  try {
    const v = fn();
    return v === undefined ? fallback : v;
  } catch {
    return fallback;
  }
}

async function safeAsync(fn, fallback = null, timeoutMs = PERMISSION_TIMEOUT_MS) {
  try {
    return await Promise.race([
      Promise.resolve().then(fn),
      new Promise((resolve) => setTimeout(() => resolve(fallback), timeoutMs)),
    ]);
  } catch {
    return fallback;
  }
}

function webgl() {
  const out = { webgl_vendor: null, webgl_renderer: null };
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    if (!gl) return out;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    out.webgl_vendor = String(gl.getParameter(dbg ? dbg.UNMASKED_VENDOR_WEBGL : gl.VENDOR)).slice(0, 128);
    out.webgl_renderer = String(gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER)).slice(0, 128);
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext(); // free the context; browsers cap how many exist
  } catch {
    /* no WebGL is itself information: both stay null */
  }
  return out;
}

function automationGlobals() {
  const found = [];
  const scan = (obj) => {
    try {
      for (const k of Object.getOwnPropertyNames(obj)) {
        if (AUTOMATION_GLOBAL.test(k)) found.push(k.slice(0, 40));
        if (found.length >= 10) return;
      }
    } catch {
      /* ignore */
    }
  };
  scan(window);
  scan(document);
  return found;
}

/** @returns {Promise<Record<string, unknown>>} sent as Sample.env */
export async function collectEnv() {
  const t0 = performance.now();
  const nav = typeof navigator !== 'undefined' ? navigator : {};
  const env = {
    probe_version: 1,
    webdriver: safe(() => nav.webdriver === true, false),
    ua: safe(() => String(nav.userAgent).slice(0, 256), ''),
    ua_brands: safe(() => (nav.userAgentData ? nav.userAgentData.brands.map((b) => String(b.brand).slice(0, 64)) : null)),
    ua_mobile: safe(() => (nav.userAgentData ? !!nav.userAgentData.mobile : null)),
    uach_platform: safe(() => (nav.userAgentData ? String(nav.userAgentData.platform).slice(0, 32) : null)),
    platform: safe(() => String(nav.platform).slice(0, 32)),
    plugins: safe(() => nav.plugins.length),
    mime_types: safe(() => nav.mimeTypes.length),
    languages: safe(() => Array.from(nav.languages || []).slice(0, 10).map((l) => String(l).slice(0, 16))),
    outer_width: safe(() => window.outerWidth),
    outer_height: safe(() => window.outerHeight),
    inner_width: safe(() => window.innerWidth),
    inner_height: safe(() => window.innerHeight),
    screen_width: safe(() => screen.width),
    screen_height: safe(() => screen.height),
    avail_width: safe(() => screen.availWidth),
    avail_height: safe(() => screen.availHeight),
    color_depth: safe(() => screen.colorDepth),
    hardware_concurrency: safe(() => nav.hardwareConcurrency),
    device_memory: safe(() => nav.deviceMemory),
    max_touch_points: safe(() => nav.maxTouchPoints),
    has_window_chrome: safe(() => typeof window.chrome === 'object' && window.chrome !== null, false),
    notification_permission: safe(() => (typeof Notification !== 'undefined' ? Notification.permission : null)),
    permissions_notifications: null,
    timezone: safe(() => Intl.DateTimeFormat().resolvedOptions().timeZone),
    automation_globals: safe(automationGlobals, []),
    ...safe(webgl, { webgl_vendor: null, webgl_renderer: null }),
  };
  env.permissions_notifications = await safeAsync(async () => {
    if (!nav.permissions || !nav.permissions.query) return null;
    const p = await nav.permissions.query({ name: 'notifications' });
    return p.state;
  });
  env.probe_ms = safe(() => Math.round((performance.now() - t0) * 10) / 10);
  return env;
}
