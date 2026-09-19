---
title: Account Holder Verification — Startups and Products
tags:
  - big-idea
  - identity
  - research/market
updated: 2026-09-18
---

# Startups and products

## Summary

This is a **mature, consolidating, well-capitalised market** — not a greenfield. Behavioural biometrics alone was valued around USD 19.5 bn in 2025, and consolidation accelerated sharply: LexisNexis Risk Solutions absorbed BehavioSec, Mastercard owns NuData, and **Visa announced acquisition of BioCatch for USD 2.4 bn in August 2026**. Anyone entering should assume the *signals* are commoditised and compete on something else: a specific vertical, an evaluation/transparency story, on-device privacy architecture, or the under-served active-challenge approach.

> [!warning] Vendor claim caveat
> Effectively none of these vendors publish reproducible error rates with stated evaluation protocols. Accuracy figures in this note are **vendor claims**, not independent validation. They establish what the market asserts is achievable, not what is achievable.

## Behavioural biometrics

| Company | Finding summary | Status | Link |
|---|---|---|---|
| **BioCatch** | The category leader. Analyses mouse, keystroke, touch and "cognitive biometrics" — including detecting when a user is being **socially engineered on a phone call while banking**, aimed at authorised-push-payment fraud. Holds the key active-perturbation patent family (see [[big_idea/04 Patents]]). | Raised USD 35 m Series E Sept 2025; **being acquired by Visa for USD 2.4 bn, announced Aug 2026**. | [biocatch.com](https://www.biocatch.com/) |
| **BehavioSec** | Stockholm origin, keystroke and pointer analytics; one of the oldest players in the category. | Acquired by and operating under **LexisNexis Risk Solutions**. | [LexisNexis Risk](https://risk.lexisnexis.com/) |
| **NuData Security** | Passive behavioural scoring for ATO, bots and scripted attacks. | Owned by **Mastercard**. | [Mastercard NuData](https://nudatasecurity.com/) |
| **TypingDNA** | Founded 2016, focused specifically on **keystroke dynamics**. Notable for offering a developer-accessible API and free tier — the most practical vendor for a research comparison baseline. | Independent. | [typingdna.com](https://www.typingdna.com/) |
| **Callsign** | Behavioural plus device plus threat intelligence, positioned around policy orchestration. | Independent. | [callsign.com](https://www.callsign.com/) |
| **Plurilock** | Continuous authentication via keystroke and mouse, with a public education-oriented technical library. | Independent, public. | [plurilock.com](https://plurilock.com/deep-dive/keystroke-dynamics/) |
| **Darwinium** | Edge-deployed (CDN-layer) behavioural and device intelligence across the full session journey rather than at login only. Architecturally the most interesting recent entrant. | Independent. | [darwinium.com](https://www.darwinium.com/) |
| **Zighra, Twosense, Neuro-ID** | Smaller players: continuous behavioural auth (Zighra, Twosense) and behavioural analytics for intent/friction measurement (Neuro-ID). | Independent. | [biometricupdate directory](https://www.biometricupdate.com/service-directory/behavioral-biometrics) |

## Device fingerprinting and device intelligence

| Company | Finding summary | Status | Link |
|---|---|---|---|
| **Fingerprint (FingerprintJS)** | The reference implementation of browser fingerprinting. The **open-source v3 library is the natural research baseline**; the commercial product adds server-side correlation and identity persistence. Honest assessment from competitors: the OSS version is spoofable client-side and lacks server correlation. | Commercial, with a widely used OSS core. | [fingerprint.com](https://fingerprint.com/blog/browser-fingerprinting-techniques/) · [GitHub](https://github.com/fingerprintjs/fingerprintjs) |
| **Castle** | **Purpose-built for account takeover specifically** — the closest commercial analogue to this project. Associates fingerprints with accounts and flags when a new fingerprint is anomalous *relative to the other devices already on that account*. That per-account relative framing is the right one. Publishes genuinely useful research on fingerprint harvesting by bots. | Independent. | [castle.io](https://castle.io/research/fingerprint-harvesting-in-the-bot-ecosystem/) · [docs](https://docs.castle.io/docs/device-fingerprinting) |
| **SEON** | Device fingerprinting combined with **digital-footprint enrichment** (does this email/phone exist on other platforms?) and a self-serve rules engine. The enrichment angle is a signal family outside the browser entirely. | Independent. | [seon.io](https://seon.io/resources/device-fingerprinting/) |
| **DataDome** | Real-time bot defence across web, mobile and API; ML models trained on large-scale traffic. Oriented to volume attacks rather than targeted individual impersonation. | Independent. | [datadome.co](https://datadome.co/) |
| **Arkose Labs** | "Arkose Device ID" — persistent fingerprinting claimed at sub-50 ms, plus adaptive challenge-based defence. Their challenge products are a commercial precedent for [[big_idea/08 Active Challenge Designs|active challenges]]. | Independent. | [arkoselabs.com](https://www.arkoselabs.com/arkose-device-id) |
| **HUMAN Security (PerimeterX)**, **F5 Distributed Cloud (Shape Security)**, **Sift**, **Socure**, **IPQualityScore**, **Incognia** | Adjacent fraud/bot/identity platforms with overlapping device-intelligence capability. Incognia is notable for **location-behaviour** fingerprinting on mobile. | Independent / acquired. | — |

## Identity platforms with built-in risk engines

| Product | Finding summary | Relevance |
|---|---|---|
| **Microsoft Entra ID Protection** | Risk signals including impossible travel, anonymous IP, unfamiliar sign-in properties, leaked credentials; drives Conditional Access step-up. | The most documented production risk-signal taxonomy available — effectively a free specification of which context signals are worth computing. |
| **Okta Adaptive MFA / Auth0 Attack Protection** | Device, network, and behaviour-based risk scoring feeding step-up policies. | Reference for policy/response design. |
| **Google reCAPTCHA Enterprise** | Returns a risk score from passive behavioural and device signals rather than a challenge. | The largest-scale deployed instance of exactly this pattern. |
| **Cloudflare Turnstile** | Privacy-preserving, non-interactive challenge; publishes its reasoning about avoiding fingerprinting. | Useful counter-model: how to get signal while explicitly minimising fingerprinting. |

## Open-source tooling (the practical build stack)

| Tool | Finding summary | Use in this project |
|---|---|---|
| **FingerprintJS v3 (OSS)** | ~30 browser attributes, mature, MIT-era licensing on v3. | **Baseline collector for [[big_idea/09 Experiment Plan|Experiment 1]].** Fork and extend rather than writing from scratch. |
| **ThumbmarkJS** | Lighter, modern, actively maintained alternative. | Comparison collector. |
| **CreepJS** | Research-grade demonstrator that specifically detects **lying/spoofed** browsers by cross-checking signals for internal inconsistency. | The best available reference for consistency-check logic. |
| **BotD** (Fingerprint) | Open-source bot detection. | Automation/RAT detection layer. |
| **JA4+ suite** (FoxIO) | TLS/HTTP client fingerprinting; successor to JA3, permissively licensed. | **Server-side fingerprint** — the unspoofable axis. |
| **SimpleWebAuthn** | Well-documented WebAuthn/passkey server and browser libraries. | Layer A implementation. |

## Synthesis for builders

> [!important] Finding
> The market has thoroughly solved *signal collection* and has **not** solved: honest published error rates, the local-stranger quadrant, graceful legitimate-change handling, or privacy-respecting architecture. Castle's per-account relative framing and Darwinium's edge deployment are the two most transferable ideas. TypingDNA and FingerprintJS OSS are the two most useful comparison baselines.

> [!note] Positioning consequence
> Competing on "better fingerprint entropy" is competing on a solved, commoditised axis. The defensible positions are **(a)** the active-challenge approach, which is under-served, **(b)** on-device/federated scoring that never transmits raw behavioural data, and **(c)** a rigorous, published evaluation protocol in a field where nobody publishes one.

## Related notes

- [[big_idea/04 Patents|Patents]]
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]
- [[big_idea/08 Active Challenge Designs|Active challenge designs]]
