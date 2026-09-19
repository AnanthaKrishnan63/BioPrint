---
title: Account Holder Verification — Landscape and Research Questions
tags:
  - big-idea
  - identity
  - research/landscape
updated: 2026-09-18
---

# Landscape and research questions

## Summary

The market has converged on a standard shape: a JavaScript collector gathers device and behavioural signals, a server fuses them with network context into a risk score, and the score drives a step-up challenge rather than a block. Every major fraud vendor sells some version of this. What remains genuinely unsolved is **not signal collection** — it is calibration at realistic base rates, graceful handling of legitimate change (new laptop, new keyboard, injury, travel), the near-total absence of reproducible cross-session error rates in vendor claims, and the fact that the highest-value quadrant (stranger on the owner's own machine) is the one with the least published evidence.

Two structural shifts define the 2026 context. First, **the JavaScript entropy surface is shrinking** — Chrome froze the User-Agent string, Firefox and Safari inject noise into canvas, audio and WebGL, and timer precision is coarsened for Spectre — while **server-side network fingerprints (TLS JA4, HTTP/2 settings) are ascendant** because JavaScript cannot touch them. Second, **cryptographic device binding finally works**: passkeys reached explicit AAL2 status in NIST SP 800-63-4, and Chrome's Device Bound Session Credentials make a stolen session cookie useless off-device. That last development undercuts the single largest real-world account-takeover vector — infostealer cookie theft — and changes what the risk-scoring layer is actually *for*.

## Current market fronts

| Front | What the market is doing | Research finding summary |
|---|---|---|
| Device fingerprinting | Canvas, WebGL/WebGPU, font probing, audio, UA Client Hints, screen geometry, hardware counts; fused into a stable device ID | Technically mature and commoditised. The hard part is no longer entropy — a full collector yields 50+ bits against the ~33 needed to single out one device globally — it is **fuzzy matching under drift**. Browsers auto-update roughly every four weeks and change canvas output, UA-CH and GPU strings. Exact-hash fingerprints have a half-life of weeks. |
| Server-side network fingerprinting | TLS ClientHello (JA3/JA4), HTTP/2 SETTINGS and pseudo-header order, header casing/order, TCP stack | Rising in relative value precisely because client JS is being restricted. Unspoofable from page JavaScript. Most useful as a **consistency check** against the claimed user agent rather than as an identifier in its own right. |
| Behavioural biometrics | Keystroke dwell/flight, mouse velocity and curvature, scroll rhythm, touch pressure, device motion | Rich academic literature with very low published error rates, but those numbers are largely **same-session** and do not survive honest cross-session, cross-device protocols. Vendors publish essentially no reproducible error rates. |
| Active challenges | Deliberately perturbed UI, scrambled keypads, reaction/correction probes, cognitive timing tasks | Small but real body of work and patents (notably BioCatch's introduced-interference family). Converts a passive, slow, low-signal problem into a fast, high-signal one, at the cost of user friction. **The least-explored quadrant of the design space.** See [[big_idea/08 Active Challenge Designs]]. |
| Network and context risk | IP, ASN, proxy/VPN/datacenter classification, geo, impossible travel, login-hour histograms | Cheap, immediate, no cold start. Long-term RBA study evidence shows IP/ASN plus device plus login-time captures most of the achievable separation for the *remote attacker* case — and none of it for the *local stranger* case. |
| Cryptographic binding | WebAuthn/passkeys, Device Bound Session Credentials, first-party storage tokens | The strongest available evidence and the cheapest to implement correctly. Underused because it needs enrolment. Should short-circuit the entire risk pipeline when present. |
| Bot and RAT detection | `isTrusted`, `navigator.webdriver`, CDP artefacts, pointer teleporting, suspiciously low timing variance | Adjacent but distinct problem. Matters here because a remote-access trojan produces a **perfect device fingerprint with alien behaviour** — exactly the hard quadrant. |

## What the browser constraint actually removes

| Not available from a browser | Consequence |
|---|---|
| OS-level hardware IDs, MAC address, serial numbers | No stable hardware anchor; everything is inference |
| Continuous camera/mic without a visible permission prompt and indicator | Face/voice biometrics are opt-in and conspicuous; unusable for passive continuous checks |
| Raw local IP | mDNS `.local` obfuscation in WebRTC since ~2020; only the public IP is reachable via STUN |
| Cross-site identity | Third-party cookies are partitioned or gone; identity is per-origin only |
| High-resolution timers | Coarsened to ~100 µs (Chrome) / 1 ms (Firefox, worse under `resistFingerprinting`) for Spectre mitigation — **directly degrades keystroke dwell-time features** |
| Installed software inventory | Only indirect proxies: fonts, codecs, TTS voices, extension resource probing |

> [!important] The constraint is less limiting than it looks
> Everything needed for both evidence axes is browser-reachable. What the constraint really removes is the ability to *anchor* identity in hardware — which is exactly what WebAuthn and DBSC restore, cryptographically, without an agent.

## High-value research questions

1. **The hard quadrant.** What is the achievable EER for distinguishing two humans *conditioned on an identical device fingerprint* — i.e. when fingerprinting contributes literally zero information? Almost nothing in the literature reports this conditional number, yet it is the entire value proposition.
2. **Honest cross-session error.** How much does behavioural EER degrade from same-session to cross-session to cross-device to cross-browser evaluation? Published figures routinely differ by a factor of 2–5 across these protocols.
3. **Timer coarsening.** How much keystroke discriminability survives quantisation to 1 ms? Are dwell-time features still viable under Firefox `resistFingerprinting`, or must the feature set shift to ratios and orderings that are quantisation-robust?
4. **Fingerprint drift half-life.** Empirically, what fraction of each attribute survives 7 / 30 / 90 days on the same device? Which minimal subset maximises uniqueness *subject to* a stability floor, rather than maximising entropy?
5. **Detection delay.** For continuous authentication, how many keystrokes or seconds of pointer activity are needed to flag an impostor at a fixed FAR of 0.1%? Detection delay, not EER, determines whether continuous auth is usable.
6. **Active vs. passive trade.** How much friction does a well-designed active challenge cost, and how many minutes of passive observation is one challenge worth?
7. **Base-rate calibration.** At a genuine impostor prevalence of roughly 1 in 10⁴–10⁶, what operating point produces an acceptable ratio of step-up challenges to true detections?
8. **Legitimate-change handling.** Can the system distinguish "new device, same person" from "new device, new person" using behaviour alone, fast enough to avoid locking out a user who just bought a laptop?
9. **Equity and accessibility.** How do screen readers, switch access, voice control, tremor, and one-handed use shift behavioural feature distributions, and what is the false-rejection cost imposed on those users?
10. **Adversarial floor.** What happens under an attacker who has *observed* the victim typing (video, shoulder-surf, keylogger replay)? Replay of captured timing is the realistic upper bound on behavioural security, and it is rarely modelled.

## Opportunity gaps supported by the landscape

- **The conditional experiment.** Deliberately collect data where the device is held constant and the person varies. This is a small, cheap study that essentially nobody publishes, and it directly answers the project's question.
- **Quantisation-robust behavioural features.** The literature was largely built before timer coarsening. Features designed for 1 ms resolution — rank orderings, ratios, distributional shape rather than absolute milliseconds — are underexplored.
- **Active challenge design as a first-class object.** Perturbation probes, scrambled keypads and reaction tests turn a 5-minute passive observation into a 5-second measurement. The patent landscape is thin outside BioCatch's family.
- **Explicit two-axis disagreement modelling.** Most systems collapse everything into one score. Reporting *which axis fired* turns an opaque risk number into an actionable diagnosis ("known device, unfamiliar hands" versus "new device, familiar hands") and directly determines the right response.
- **Honest evaluation harness.** A public, reproducible protocol with cross-session splits and realistic base rates would be a genuine contribution given how unreliable published numbers are.

## Related notes

- [[big_idea/02 Papers|Papers]]
- [[big_idea/03 Startups and Products|Startups and products]]
- [[big_idea/04 Patents|Patents]]
- [[big_idea/05 Datasets and Data Availability|Datasets and data availability]]
- [[big_idea/06 Frameworks and Validation|Frameworks and validation]]
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]
