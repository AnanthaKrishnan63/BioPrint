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
//
// probe_version 2 adds the Panopticlick attributes engine/device.py compares against
// enrollment (OWNER of those keys: device agent). Each is wrapped, cheap and may be null:
//     fonts            sorted list of the FONT_CANDIDATES present, by width measurement
//     cookies_enabled  navigator.cookieEnabled
//     do_not_track     navigator.doNotTrack, normalised to a string or null
//     canvas_hash      8-hex FNV-1a of a fixed text+shape drawing. Firefox (RFP) and
//                      Safari/Brave may add noise: device.py then sees no majority value
//                      across enrollment and ignores it, rather than us guessing here.
//     audio_hash       OfflineAudioContext oscillator -> compressor, summed samples.
//                      Optional: null if it does not settle inside AUDIO_TIMEOUT_MS.

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

// ---------------------------------------------------------------- device attributes (probe_version 2)
const AUDIO_TIMEOUT_MS = 40;

// Common fonts across Windows, macOS and Linux desktops. Presence is decided by
// measuring a test string in "'Candidate', <fallback>" against the bare fallback:
// if any of the three fallbacks gives a different width or height, the candidate
// exists. The list is fixed on purpose: the set of PRESENT fonts is the attribute.
const FONT_CANDIDATES = [
  'Arial', 'Arial Black', 'Arial Narrow', 'Calibri', 'Cambria', 'Comic Sans MS', 'Consolas',
  'Courier New', 'Georgia', 'Impact', 'Lucida Console', 'Lucida Sans Unicode', 'Microsoft Sans Serif',
  'Palatino Linotype', 'Segoe UI', 'Tahoma', 'Times New Roman', 'Trebuchet MS', 'Verdana', 'Wingdings',
  'Helvetica', 'Helvetica Neue', 'Menlo', 'Monaco', 'Geneva', 'Optima', 'Futura', 'Gill Sans',
  'Baskerville', 'American Typewriter', 'Avenir', 'Hiragino Sans', 'Apple Color Emoji',
  'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono', 'Liberation Sans', 'Liberation Serif',
  'Liberation Mono', 'Ubuntu', 'Ubuntu Mono', 'Cantarell', 'Noto Sans', 'Noto Serif', 'Noto Color Emoji',
  'Roboto', 'Droid Sans', 'FreeSans', 'Nimbus Sans', 'Fira Sans', 'Open Sans', 'Inter', 'Source Sans Pro',
];
const FONT_FALLBACKS = ['monospace', 'sans-serif', 'serif'];
const FONT_TEXT = 'mmmMMMwwwlli10Oo@#';

/** @returns {string[] | null} sorted list of candidates that are installed */
function detectFonts() {
  if (fontsCache === undefined) {
    const r = measureFonts();
    if (r) fontsCache = r; // null (no body yet) is not cached, so a later call retries
    return r;
  }
  return fontsCache;
}

// Measured (Chromium on Linux): the FIRST reference to each font family costs ~5 ms of
// fontconfig lookup, so a cold measurement is ~160 ms; warm it is ~5 ms. The user
// spends seconds typing a password, so we warm up while the page is idle and
// collectEnv() reads the cached answer. A submit that beats the idle callback pays the
// cold cost once (that is a script, not a person).
let fontsCache;
const idle = typeof requestIdleCallback === 'function' ? requestIdleCallback : (fn) => setTimeout(fn, 0);
if (typeof document !== 'undefined') {
  idle(() => {
    if (fontsCache === undefined) safe(detectFonts);
  });
}

function measureFonts() {
  if (typeof document === 'undefined' || !document.body) return null;
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:-9999px;top:0;visibility:hidden;font-size:72px;line-height:normal;white-space:nowrap';
  const span = (family) => {
    const s = document.createElement('span');
    s.style.fontFamily = family;
    s.textContent = FONT_TEXT;
    host.append(s);
    return s;
  };
  const base = FONT_FALLBACKS.map(span);
  const probes = FONT_CANDIDATES.map((f) => FONT_FALLBACKS.map((fb) => span(`'${f}', ${fb}`)));
  document.body.append(host);
  try {
    // All spans are in the DOM before the first read, so this is one layout pass.
    const size = (s) => `${s.offsetWidth}x${s.offsetHeight}`;
    const baseSize = base.map(size);
    const present = [];
    FONT_CANDIDATES.forEach((f, i) => {
      if (probes[i].some((s, j) => size(s) !== baseSize[j])) present.push(f);
    });
    return present.sort();
  } finally {
    host.remove();
  }
}

/** FNV-1a over a string, as 8 hex characters. Not a security hash, just a short label. */
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

/** @returns {string | null} hash of a fixed drawing; null when canvas is unavailable */
function canvasHash() {
  const c = document.createElement('canvas');
  c.width = 240;
  c.height = 60;
  const ctx = c.getContext('2d');
  if (!ctx) return null;
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#f60';
  ctx.fillRect(100, 2, 60, 20);
  ctx.fillStyle = '#069';
  ctx.font = '15px Arial';
  ctx.fillText('BioPrint, <canvas> 1.0 @#%', 4, 18);
  ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
  ctx.font = '17px Times New Roman';
  ctx.fillText('BioPrint, <canvas> 1.0 @#%', 6, 40);
  ctx.globalCompositeOperation = 'multiply';
  for (const [color, x] of [['#f2f', 60], ['#2ff', 90], ['#ff2', 75]]) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, 40, 15, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(180, 10);
  ctx.bezierCurveTo(200, 50, 220, 5, 236, 55);
  ctx.stroke();
  return fnv1a(c.toDataURL());
}

/** @returns {Promise<string | null>} summed compressor output of a fixed oscillator */
async function audioHash() {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) return null;
  const ctx = new Ctx(1, 5000, 44100);
  const osc = ctx.createOscillator();
  osc.type = 'triangle';
  osc.frequency.value = 10000;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -50;
  comp.knee.value = 40;
  comp.ratio.value = 12;
  comp.attack.value = 0;
  comp.release.value = 0.25;
  osc.connect(comp);
  comp.connect(ctx.destination);
  osc.start(0);
  const buf = await ctx.startRendering();
  const data = buf.getChannelData(0);
  let sum = 0;
  for (let i = 4500; i < 5000; i++) sum += Math.abs(data[i]);
  return sum.toFixed(5);
}

function doNotTrack() {
  const nav = navigator;
  const v = nav.doNotTrack ?? window.doNotTrack ?? nav.msDoNotTrack;
  return v == null ? null : String(v).slice(0, 16);
}

/** @returns {Promise<Record<string, unknown>>} sent as Sample.env */
export async function collectEnv() {
  const t0 = performance.now();
  const nav = typeof navigator !== 'undefined' ? navigator : {};
  const env = {
    probe_version: 2,
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
    // ---- probe_version 2: device attributes (engine/device.py)
    fonts: safe(detectFonts),
    cookies_enabled: safe(() => (typeof nav.cookieEnabled === 'boolean' ? nav.cookieEnabled : null)),
    do_not_track: safe(doNotTrack),
    canvas_hash: safe(canvasHash),
    audio_hash: null,
  };
  // The two async probes race their own timeouts, side by side, so neither holds up login.
  [env.permissions_notifications, env.audio_hash] = await Promise.all([
    safeAsync(async () => {
      if (!nav.permissions || !nav.permissions.query) return null;
      const p = await nav.permissions.query({ name: 'notifications' });
      return p.state;
    }),
    safeAsync(audioHash, null, AUDIO_TIMEOUT_MS),
  ]);
  env.probe_ms = safe(() => Math.round((performance.now() - t0) * 10) / 10);
  return env;
}
