---
title: Browser Signal Catalogue
tags:
  - big-idea
  - identity
  - research/technical
updated: 2026-09-18
---

# Browser signal catalogue

The complete inventory of what is reachable under the browser-only constraint, organised by the four evidence layers. Entropy figures are approximate, drawn from Cover Your Tracks / AmIUnique-style measurements over a general web population.

> [!warning] This note ages fastest
> Browser fingerprinting surfaces change every release cycle. Chrome froze the User-Agent string, Firefox and Safari inject noise into canvas/audio/WebGL, and Privacy Sandbox was wound down in 2025 leaving third-party cookies in place but the JS entropy surface still reduced. **Re-verify any single API before depending on it.**

---

## Layer A — Cryptographic binding

The strongest layer, and the one most projects skip. Highest value per unit of effort.

| Mechanism | What it proves | Availability | Notes |
|---|---|---|---|
| **WebAuthn / passkeys** — `navigator.credentials.get()` | Possession of a private key **plus** local user verification (biometric or PIN) | Universal | Platform authenticators are TPM/Secure-Enclave bound and non-exportable. `PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable()` tells you whether to offer it. Conditional UI ("autofill") makes it near-frictionless. NIST SP 800-63-4 admits syncable passkeys at **AAL2**. |
| **Device Bound Session Credentials (DBSC)** | The session cookie is cryptographically bound to a TPM key — **a stolen cookie is useless off-device** | Chrome, rolling out | Directly defeats infostealer cookie theft, which is the dominant real-world ATO vector. Highest-leverage single thing to implement. |
| **First-party storage token** — `localStorage`, IndexedDB, Cache API, service-worker-held secret | "This browser profile has been here before" | Universal | Cheap, survives IP change. Destroyed by clearing data, incognito, profile switch. **Use as a positive signal only** — absence must never itself be suspicious. |
| **Partitioned cookies (CHIPS)**, Storage Access API | Same, across embedded contexts | Modern browsers | Cross-site identity is effectively dead; assume per-origin only. |

> [!important] Architectural rule
> A valid device-bound credential should **short-circuit almost all risk scoring**. Layers B, C and D exist to score the sessions that *don't* have one, and to decide when to demand one.

---

## Layer B — Device and browser fingerprint

### B.1 Client-side JavaScript signals

| Signal | API | ~Entropy (bits) | Stability | Spoofability | Notes |
|---|---|---|---|---|---|
| **Canvas render hash** | Draw text + shapes → `toDataURL()` | 8–14 | High until GPU/driver update | Medium | Highest single contributor. Safari and Firefox inject per-session noise; **average several draws** or detect the noise rather than treating it as signal. |
| **Font list** | `document.fonts.check()` probing, or `queryLocalFonts()` | 8–14 | High | Low–medium | Measurement-based probing works everywhere. `queryLocalFonts()` (Chromium) gives a full list but **triggers a permission prompt** — usually not worth it. |
| **WebGL / WebGPU renderer** | `WEBGL_debug_renderer_info`, `navigator.gpu.requestAdapter()` → `adapter.info` | 6–12 | Very high | Medium | Chromium exposes real GPU strings; Firefox restricts under `resistFingerprinting`; Safari returns generic values. WebGPU `adapter.info` is the newer, richer surface. |
| **Audio fingerprint** | `OfflineAudioContext` → oscillator → `DynamicsCompressor` → sum output | 4–8 | High | Medium | Reflects the audio DSP implementation, which varies by CPU/OS. Noised in Firefox/Safari. |
| **Screen geometry** | `screen.width/height/availWidth/colorDepth`, `devicePixelRatio`, `outerHeight − innerHeight` | 4–8 | Medium | High | `availWidth` leaks taskbar/dock position. The outer−inner delta leaks browser chrome, zoom level and some extensions. Changes with external monitors. |
| **UA Client Hints (high entropy)** | `navigator.userAgentData.getHighEntropyValues(['platformVersion','model','architecture','bitness','fullVersionList'])` | 5–10 | Medium — OS/browser updates | High | **Chromium only.** The replacement for the frozen UA string. Must be explicitly requested. |
| **Timezone and locale** | `Intl.DateTimeFormat().resolvedOptions()`, `navigator.languages`, `Date.getTimezoneOffset()` | 3–6 | High | High | Low entropy alone, but the **coherence check** against Geo-IP is high value. Firefox RFP spoofs timezone to UTC. |
| **Hardware counts** | `navigator.hardwareConcurrency`, `navigator.deviceMemory`, `navigator.storage.estimate()` | 2–5 | Very high | High | `deviceMemory` is Chromium-only; Safari caps `hardwareConcurrency`. Storage quota is heavily quantised now. |
| **Codec / media capability matrix** | `MediaCapabilities.decodingInfo()`, `MediaRecorder.isTypeSupported()` over a codec list | 3–6 | High | Low | Reflects OS codec installation and hardware decode support. Underused and hard to fake convincingly. |
| **TTS voice list** | `speechSynthesis.getVoices()` | 4–8 | High | Low | Strongly OS- and locale-dependent. Genuinely good signal; needs an async settling delay before reading. |
| **Media device topology** | `navigator.mediaDevices.enumerateDevices()` | 2–4 | Medium | Medium | Without permission you get **counts and kinds only** — labels are hidden and `deviceId` is salted per origin. The count vector alone is still useful. |
| **Math / float quirks** | `Math.tanh`, `Math.expm1`, etc. at extreme inputs | 1–3 | Very high | Low | Reveals the libm/JS engine build. Tiny entropy, near-zero cost, very stable. |
| **ClientRects** | `getClientRects()` on styled text | 2–4 | High | Low | Sub-pixel text layout differs by font-rendering stack. |
| **Extension detection** | Web-accessible-resource probing, DOM-mutation detection | 2–10 | Medium | Low | Potentially high entropy but fragile and ethically questionable. Detecting an *ad blocker* is much less invasive than enumerating everything. |
| **Battery** | `navigator.getBattery()` | ~0 static | — | Medium | Removed in Firefox and Safari; Chromium only. Static value is useless, but the **level/charging curve links sessions within a short window**. |
| **Performance benchmark** | Timed micro-benchmark → CPU/GPU class | 2–4 | Medium | Low | Noisy and slow. Use only as a coarse tier, never as an identifier. |
| **Display refresh rate + frame jitter** | `requestAnimationFrame` interval statistics | 2–4 | Very high | Low | 60 / 90 / 120 / 144 / 165 Hz, plus GPU/compositor jitter signature. Cleanly separates an internal 120 Hz laptop panel from an external 60 Hz monitor. Underused; costs ~500 ms of sampling. See [[big_idea/08 Active Challenge Designs\|A8b]]. |

**Combined: 50+ bits**, against the ~33 needed to single out one device among 8.5 billion.

> [!important] Do not maximise entropy
> In a realistic application population — thousands to millions of users — you have enormous headroom. **Deliberately downsample to stable, low-churn attributes.** Over-fitting the fingerprint is the single largest cause of false "new device" alerts, because browsers auto-update roughly monthly and change canvas output, UA-CH and GPU strings.

### B.2 Server-side signals (free, unspoofable from JavaScript)

These arrive with every request. Page JavaScript cannot touch them, so an attacker using curl, Python or a patched headless stack fails them immediately.

| Signal | What it captures |
|---|---|
| **TLS ClientHello → JA3 / JA4** | Cipher suites, extensions and their order, GREASE values, ALPN, supported groups |
| **HTTP/2 SETTINGS + priority + pseudo-header order** | The "Akamai H2 fingerprint"; distinctive per HTTP stack |
| **HTTP header order and casing** | Browsers have fixed, recognisable header orders; libraries do not |
| **TCP/IP stack** | Initial TTL, window size → passive OS inference |

> [!important] The single highest-value check in the whole system
> **Does the JA4 fingerprint agree with the claimed UA / UA-CH?** A "Chrome 141 on macOS" user agent arriving with a Python `ssl` JA4 is a hard fail. This one consistency check is worth more than any individual JavaScript signal, and it costs nothing at runtime.

### B.3 Fuzzy matching, not exact matching

**Never hash all attributes into one ID.** Fingerprints drift constantly.

```
1. Store the attribute vector, not a hash.
2. Weight each attribute by (entropy × stability),
   where stability = P(unchanged | same device, 30 days), measured empirically.
3. Score a candidate against stored device records with weighted similarity.
4. Bands:  > 0.85  → same device
           0.60–0.85 → device evolved; UPDATE the stored record
           < 0.60  → new device
5. Keep per-device HISTORY (a list of observed vectors) so gradual drift chains correctly.
```

---

## Layer C — Network and context

| Signal | Use | Caveat |
|---|---|---|
| IP, ASN, ISP class (residential / mobile / datacenter / VPN / Tor) | Datacenter ASN on a consumer account is a strong risk bump | VPN use is normal and increasing; do not treat as guilt |
| Geo-IP city/country | Coarse location history | Mobile carrier NAT places users hundreds of km away |
| **Impossible travel** — Δdistance / Δtime > ~1000 km/h | Classic, still effective | Must whitelist VPN and carrier-NAT patterns or it fires constantly |
| **Timezone ↔ Geo-IP coherence**; `navigator.languages` ↔ country | Mismatch is a proxy/automation tell | Legitimate for expats, travellers, multilingual users |
| **ASN / subnet familiarity** — has this account used this /24 or ASN before? | **Far better than exact IP match** | The most useful context feature per the long-term RBA study |
| Login hour and weekday histogram per account | A 03:00 login for a strict 09:00–17:00 user is real signal | Needs ~20+ prior logins to be meaningful |
| Session cadence — inter-login interval, duration, request rate | Credential-stuffing and scripted access | — |
| Entry path — direct-to-login vs. normal navigation | Automation tell | — |

---

## Layer D — Behavioural biometrics

The **only** layer that answers "who is at the keyboard", and therefore the only defence for the hard quadrant.

> [!warning] Behavioural templates must be stored **per (user, device class)**
> Typing speed and pointer dynamics are properties of *the person on a particular device*, not of the person alone. A mechanical keyboard, a laptop chiclet keyboard and a touchscreen produce three different distributions for the same human. Comparing a session on one device class against a template built on another will false-reject almost every time — this is a large part of the gap between published lab EERs and production performance. See [[big_idea/08 Active Challenge Designs|A7]].

### D.1 Keystroke dynamics

Collected from `keydown` / `keyup`. **Use `event.code`, never `event.key` or the character.**

| Feature | Definition |
|---|---|
| **Dwell time** | Key hold duration (down → up). The single most discriminative primitive. |
| **Flight time** | Up of key *n* → down of key *n+1*. Can be **negative** (rollover) — negative flights are highly personal. |
| **Digraph / trigraph latency** | Down-down latency for frequent pairs (`th`, `in`, `er`, `he`) and, for fixed text, the actual pairs in the string |
| **Rate and burst structure** | Typing speed, pause-length distribution, burst segmentation |
| **Error behaviour** | Backspace rate, correction latency, strategy (delete-one vs. delete-word vs. select-and-retype) |
| **Modifier chording** | Shift-hold overlap, capitalisation timing, Ctrl/Alt combination rhythm |
| **Input-method habits** | Number row vs. numpad, arrow keys vs. mouse for navigation |
| **Overlap degree** | How often 2, 3 or more keys are held simultaneously. Measured at **32% of the session** for this project's first subject, against a mean dwell of 189 ms — both far from population norms, which is what a discriminative feature looks like. See [[big_idea/12 Measurements\|Measurements]]. |

**Two regimes:**
- **Fixed text** (a known passphrase/password): high accuracy, EER roughly 3–10% from a single 10-character sample with classic distance metrics, much lower with 10+ enrolled samples. The CMU benchmark setting.
- **Free text** (anything typed during a session): harder, needs ~300–1000 keystrokes to stabilise, but is what enables *continuous* authentication.

> [!warning] Timer coarsening — real, but measured smaller than feared
> `event.timeStamp` and `performance.now()` are coarsened for Spectre mitigation: **~100 µs in Chrome, 1 ms in Firefox, worse under `resistFingerprinting`.** **Test every feature at 1 ms resolution, not just in your own Chrome.**
>
> **Measured on this project's own setup** (Firefox 155, Ubuntu/X11) against kernel timestamps: quantisation is exactly 1.0 ms as expected, but dwell times still land within **2.8 ms at the 95th percentile** of the kernel's, or 1.4% of a typical dwell, with no systematic bias. On these numbers coarsening is not a threat to dwell or flight. Full results in [[big_idea/12 Measurements]].
>
> Prefer quantisation-robust constructions anyway — rank orderings, ratios between timings, distributional shape rather than absolute milliseconds — because this is one machine and one browser, and the picture may differ elsewhere.

> [!warning] Browser event streams are not perfectly faithful
> Measured on the same setup: Firefox emitted **phantom keydown events** — the same key reported down twice with no release between, ~5 ms before the genuine press — in about **1% of events**, all at points where keys overlapped within the 1 ms tick. Not OS auto-repeat, and not filtered by `event.repeat`. Any pairing logic must tolerate a duplicate keydown, and dwell measured from the first of the pair is inflated by the gap. See [[big_idea/12 Measurements]]. No mention of this behaviour was found in the surveyed literature.

### D.2 Mouse and pointer dynamics

From `pointermove` plus **`getCoalescedEvents()`**, which recovers sub-frame resolution that rAF throttling otherwise discards.

| Feature | Notes |
|---|---|
| Velocity and **acceleration** profiles | Position of peak velocity within a movement is personal |
| Path **curvature** and straightness ratio | Path length ÷ straight-line distance |
| Jitter / micro-tremor spectrum | FFT of small-scale deviation |
| Sub-movement segmentation | How a reach decomposes into ballistic + corrective phases |
| **Fitts's law regression** | Fit `MT = a + b·log₂(D/W + 1)` per user. **The (a, b) pair is remarkably personal** and is a compact, interpretable two-number signature |
| Click dwell, double-click interval | Down→up duration; inter-click gap |
| Overshoot and correction | Distance past target before settling |
| Drag dynamics | Hold-and-move differs from move-then-click |
| Scroll | Wheel delta quantum (hardware!), momentum/flick pattern, scroll-then-pause rhythm |

### D.3 Touch and motion (mobile)

| Feature | API | Notes |
|---|---|---|
| Pressure, contact radius, rotation | `Touch.force`, `radiusX/Y`, `rotationAngle` | `force` needs supporting hardware; Safari exposes it on capable devices |
| Swipe velocity, curvature, multi-touch spacing | `touchmove` | Multi-touch spacing is a crude hand-geometry proxy |
| **Hand tremor, gait, phone tilt while typing** | `devicemotion` / `deviceorientation` | **Extremely discriminative.** Requires `DeviceMotionEvent.requestPermission()` on iOS 13+ and a secure context |

> [!note] Mobile inverts the picture
> Mobile keyboards give almost no usable keystroke timing — autocorrect, swipe input and predictive text destroy it. Mobile devices are also *less* fingerprintable than desktops (fewer configuration variations). On mobile, lean on **motion sensors and touch dynamics**; on desktop, lean on keystroke and pointer.

### D.4 Interaction-level (cheap, robust, underrated)

| Feature | Why it matters |
|---|---|
| Navigation path n-grams | Which pages, in what order — habitual and hard to fake without knowing the victim |
| Time-on-page distribution | Familiar users move faster |
| **Form-fill order, and paste vs. type vs. autofill per field** | **Attackers paste stolen credentials; owners autofill or type them.** Very high signal-to-cost ratio |
| Copy/paste frequency; keyboard shortcut vs. menu preference | Stable habit |
| Zoom level, window size | Sticky per user |
| Scroll depth before acting | Familiar users scroll less |

### D.5 Remote-access and replay detection

Catches RATs and scripted sessions — the attacks that produce a **perfect device fingerprint with alien behaviour**.

| Check | Detects |
|---|---|
| Pointer "teleporting" — large coordinate jumps with no interpolation | Remote desktop, synthetic events |
| Integer-only coordinates on a fractional-DPR display | Synthetic injection |
| **Suspiciously low timing variance** | Replayed or scripted input is *too* regular. Test the coefficient of variation against a human floor |
| `event.isTrusted === false` | Programmatically dispatched events |
| Low pointer event rate | RDP/VNC compress motion |
| `navigator.webdriver`, CDP artefacts, `Permissions.query` anomalies | Headless automation |

---

## Recommended minimal combination

The concrete answer to "what unique combination of features".

> [!important] Tier 1 — always collect (solves both *different device* quadrants)
> Passkey/DBSC presence → first-party storage token → **fuzzy device fingerprint** over `{canvas hash, WebGL/WebGPU renderer, font-probe bitmap, audio hash, screen+DPR, timezone, languages, hardwareConcurrency, UA-CH platformVersion}` → **server-side JA4 + HTTP/2 fingerprint** → **ASN/subnet familiarity + impossible travel + login hour**.
> **~12 features. Immediate, no cold start, no consent complications beyond ordinary disclosure.**

> [!important] Tier 2 — adds the hard quadrant (*stranger on the owner's own machine*)
> **28-dimensional fixed-text keystroke vector captured on the login form itself** (free — they are already typing) + **Fitts's (a, b) + velocity profile + click dwell** from pointer + **paste-vs-type on credential fields** + `isTrusted` / teleport checks for RAT detection.
> **Requires enrolment over 5–30 sessions and triggers GDPR Art. 9 obligations.**

Tier 1 without Tier 2 cannot tell the account holder from the person at their unlocked desk. Tier 2 without Tier 1 has no cold-start story. **Both are required.**

## Related notes

- [[big_idea/06 Frameworks and Validation|Frameworks and validation]] — how to fuse these into a decision
- [[big_idea/08 Active Challenge Designs|Active challenge designs]] — how to get Layer D signal fast
- [[big_idea/09 Experiment Plan|Experiment plan]]
