---
title: Big Idea — Is It Really the Account Holder?
aliases:
  - Continuous Identity Assurance
  - Browser-Only Account Holder Verification
tags:
  - research/hub
  - big-idea
  - identity
  - behavioral-biometrics
  - device-fingerprinting
updated: 2026-09-18
---

# Big idea: detecting whether the account holder — or someone else — is using the session

**Constraint:** only data reachable from a **web browser** (client JavaScript plus whatever the browser's own network request reveals to the server). No OS agent, no native app, no external hardware.

## Navigate

> [!tip] New to this field?
> Start with [[big_idea/10 Basics and Glossary|Basics and glossary]] — every acronym and concept used below, explained from scratch, plus a reading order.

- [[big_idea/10 Basics and Glossary|**Basics and glossary — start here**]]
- [[big_idea/01 Landscape and Research Questions|Landscape and research questions]]
- [[big_idea/02 Papers|Relevant papers]]
- [[big_idea/03 Startups and Products|Startups and products]]
- [[big_idea/04 Patents|Representative patents]]
- [[big_idea/05 Datasets and Data Availability|Datasets and data availability]]
- [[big_idea/06 Frameworks and Validation|Frameworks and validation]]
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]] ← the core technical inventory
- [[big_idea/08 Active Challenge Designs|Active challenge designs]] ← builds on [[yooooooooo]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
- [[big_idea/11 System Design|System design]] ← signup to login, end to end
- [[big_idea/12 Measurements|Measurements]] ← what has actually been measured
- [[big_idea/13 Roadmap|Roadmap]] ← where we are, where this ends, and the decision gates

> [!summary] Executive finding
> **"Different person" and "different computer" are two separate detections, and no single signal family covers both.** Device fingerprinting and network context identify the *machine*; they are structurally blind to a stranger sitting at the owner's own unlocked laptop. Behavioural biometrics — keystroke, pointer, touch, motion — is the only browser-reachable evidence about *who is at the keyboard*, but it has no cold-start story and degrades badly on mobile. The defensible design is **two orthogonal evidence axes fused into one risk score**, with cryptographic device binding (passkeys, device-bound session credentials) short-circuiting the whole pipeline whenever it is available. The most informative feature in the system is not any single signal — it is the **disagreement pattern between the two axes**.

## The 2×2 that defines the problem

|                      | **Same device**                                                                                                                                                 | **Different device**                                                                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Same person**      | Normal login. Baseline case.                                                                                                                                    | New laptop, new phone, travel, browser reinstall. Device axis fires, behaviour axis agrees. **Must not be blocked** — this is the dominant source of false alarms. |
| **Different person** | Shared/unlocked PC, household member, shoulder-surfer, remote-access trojan, session-cookie replay. Device axis sees a perfect match. **Only behaviour fires.** | Classic credential theft from the attacker's own machine. Both axes fire. Easiest case.                                                                            |

> [!important] Design consequence
> A system built only on fingerprinting cannot distinguish the account holder from the person sitting at their desk. A system built only on behaviour cannot make a decision on the user's very first interaction. Each axis alone leaves one quadrant undefended.

## Four evidence layers

| Layer                                                                                              | Question it answers                                                  | Strength                                          | Cold start          | Spoof resistance                       |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- | ------------------- | -------------------------------------- |
| **A. Cryptographic binding** — passkeys/WebAuthn, DBSC, first-party storage tokens                 | "Does this browser hold a secret only the enrolled device can hold?" | Decisive when present                             | Requires enrolment  | Very high (hardware-backed)            |
| **B. Device fingerprint** — canvas, WebGL/WebGPU, fonts, audio, UA-CH, plus server-side TLS/HTTP-2 | "Is this the same machine and browser build?"                        | High                                              | Immediate           | Medium (client), high (server-side)    |
| **C. Network and context** — IP, ASN, geo, timezone coherence, login hour                          | "Does the circumstance match this account's history?"                | Medium                                            | Immediate           | Low–medium (VPNs are cheap)            |
| **D. Behavioural biometrics** — keystroke, pointer, touch, motion, interaction habits              | "Is this the same human?"                                            | Medium–high, the only answer to the hard quadrant | Needs 5–30 sessions | Medium (hard to mimic, easy to replay) |

## How to read the evidence

- **Paper:** peer-reviewed or preprint research. Preprints are explicitly marked.
- **Standard:** specification, RFC, or standards-body guidance (W3C, FIDO, NIST).
- **Vendor claim:** useful market evidence, not independent validation. Behavioural-biometrics vendors publish almost no reproducible error rates.
- **Patent status:** copied from Google Patents as a landscape signal, not a legal opinion.
- **Open:** direct public download. **Request:** application or agreement required.

> [!note] Snapshot date
> Browser API availability, vendor status, and patent status were checked as a research snapshot on **2026-09-18**. Browser fingerprinting surfaces change every release cycle — re-check the [[big_idea/07 Browser Signal Catalogue|signal catalogue]] before building on any single API.

> [!warning] Legal position, stated up front
> Behavioural data processed to identify a specific individual is **special-category biometric data under GDPR Art. 9** and is regulated under India's **DPDP Act 2023** and Illinois **BIPA**. This is not a note-at-the-end concern; it constrains the architecture. See [[big_idea/06 Frameworks and Validation|Frameworks and validation]].
