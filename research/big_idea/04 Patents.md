---
title: Account Holder Verification — Patents
tags:
  - big-idea
  - identity
  - research/patents
updated: 2026-09-18
---

# Representative patents

## Summary

The patent landscape is **dense and old**. Core browser fingerprinting was claimed around 2011–2015 (BlueCava and others) and behavioural biometrics from mouse/keyboard even earlier — 2005–2006 filings already claim user profiling from motion-based input devices. BioCatch holds the most aggressive and specific portfolio, and critically **already claims the active-perturbation idea**: deliberately injecting input/output interference to elicit a measurable corrective reaction from the user.

> [!warning] Patent status is a landscape signal, not a legal opinion
> Statuses and assignees below are copied from Google Patents as a research snapshot on 2026-09-18. Assignees change through acquisition and reassignment. Anything with commercial intent needs a proper freedom-to-operate opinion from counsel. Several assignee attributions below are marked *unverified* and were inferred from claim language and naming convention only.

## Device and browser fingerprinting

| Patent | Claim summary | Assignee | Link |
|---|---|---|---|
| **US8601109B2** — Incremental browser-based device fingerprinting | Building a device fingerprint **incrementally** — collecting attributes progressively rather than in one pass, to balance latency against confidence. Directly relevant to any staged collector design. | BlueCava (later reassigned; ALC → Adstra → JPMorgan Chase as security interest) | [Google Patents](https://patents.google.com/patent/US8601109) |
| **US8954560B2** — Incremental browser-based device fingerprinting | Continuation in the same family. | BlueCava lineage | [Google Patents](https://patents.google.com/patent/US8954560B2/en) |
| **US9942349** — Incremental browser-based device fingerprinting | Later continuation; the family spans roughly 2011–2018. | **BlueCava, Inc.** (verified) | [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9942349) |
| **US20120215896A1** | Published application in the same incremental-fingerprinting family. | BlueCava lineage | [Google Patents](https://patents.google.com/patent/US20120215896) |
| **EP3039598A1** — Web browser fingerprinting | European filing on browser fingerprinting. Relevant because it establishes EP-side coverage, which matters more than US coverage given GDPR-driven deployment. | *unverified* | [Google Patents](https://patents.google.com/patent/EP3039598A1/en) |
| **US20150006384A1** — Device fingerprinting | Broad device-fingerprinting application. | *unverified* | [Google Patents](https://patents.google.com/patent/US20150006384A1/en) |
| **US20230231884A1** — Browser fingerprinting and control for session protection and private application protection | Recent (2023) and closest to this project's framing: fingerprinting used specifically for **session protection**, not advertising. Worth reading in full. | *unverified* | [Google Patents](https://patents.google.com/patent/US20230231884A1/en) |

## Behavioural biometrics — foundational

| Patent | Claim summary | Assignee | Link |
|---|---|---|---|
| **US20050008148A1** — Mouse performance identification | 2005 filing. Identifying a user from mouse performance characteristics. One of the earliest claims in the space; useful as prior art. | *unverified* | [Google Patents](https://patents.google.com/patent/US20050008148) |
| **US20060224898A1** — Determining a computer user profile from a motion-based input device | Builds a user profile from mouse/keyboard motion input. Broad and early. | *unverified* | [Google Patents](https://patents.google.com/patent/US20060224898A1/en) |
| **US9531710B2** — Behavioral authentication system using a biometric fingerprint sensor and user behavior | Fuses a hardware biometric with behavioural signals. Out of scope for browser-only, but establishes the fusion claim. | *unverified* | [Google Patents](https://patents.google.com/patent/US9531710B2/en) |
| **US20160259924A1** — Behavioural biometric authentication using program modelling | Models the *program interaction sequence* rather than raw input dynamics — the interaction-level signal family. | *unverified* | [Google Patents](https://patents.google.com/patent/US20160259924A1/en) |
| **US20210264003A1** — Keyboard and mouse based behavioral biometrics to enhance password-based login authentication using ML | Collects keystroke and mouse biometrics **during login specifically** and uses an ML model to strengthen password authentication. Very close to the Tier-2 design in [[big_idea/06 Frameworks and Validation]]. | *unverified* | [Google Patents](https://patents.google.com/patent/US20210264003A1/en) |

## BioCatch family — the important one

| Patent | Claim summary | Assignee | Link |
|---|---|---|---|
| **US10298614B2** — System, device, and method of generating and managing behavioral biometric cookies | **Claims intentionally-introduced input/output interference to elicit user reactions and extract user-specific behavioural features from the corrective mouse-pointer movement.** This is the active-perturbation idea. | **BIOCATCH LTD** (verified) | [Google Patents](https://patents.google.com/patent/US10298614B2/en) |
| **US20180034850A1** — (same title, published application) | Application in the same family. | BioCatch | [Google Patents](https://patents.google.com/patent/US20180034850A1/en) |
| **US10474815B2** — System, device, and method of detecting malicious automatic script and code injection | Transparent behavioural-biometric methods from mouse, keyboard and touch used to detect scripted/automated interaction. The bot and RAT detection branch. | BioCatch *(naming convention consistent; unverified)* | [Google Patents](https://patents.google.com/patent/US10474815B2/en) |
| **US20160180083A1** — Method, computer program and system that uses behavioral biometric algorithms | Verifying that a device is operated by a human, across banking, database and platform contexts. | *unverified* | [Google Patents](https://patents.google.com/patent/US20160180083A1/en) |

## Implications for this project

> [!important] The active-challenge idea is already claimed
> The strongest idea in [[yooooooooo|the original brainstorm]] — *"trick the user: test reaction and correction patterns"* — maps almost exactly onto **US10298614B2**. This does not kill the direction, but it changes how to approach it:
> - **For research and coursework**: no issue. Patents restrict commercial practice, not study, publication, or teaching.
> - **For a product**: the specific claim is pointer-interference-based. Other probe modalities from the brainstorm — **scrambled keypad layouts, mental-arithmetic timing via captcha, two-factor-code entry latency** — are different mechanisms and plausibly outside this family, but that is exactly the question for a freedom-to-operate search.
> - **For positioning**: a patent this specific is evidence the idea *works*. A large vendor spent money claiming it.

> [!note] Where the landscape is thin
> Searches surfaced little covering: **scrambled/randomised input layouts as an identity probe**, **cognitive-task timing (arithmetic, reaction) as a behavioural feature**, and **explicit two-axis device-vs-person disagreement modelling**. These are the most defensible directions, and they are also the most novel research contributions.

## Related notes

- [[big_idea/03 Startups and Products|Startups and products]]
- [[big_idea/08 Active Challenge Designs|Active challenge designs]]
- [[yooooooooo|Original brainstorm]]
