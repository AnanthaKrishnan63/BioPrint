---
title: Roadmap — From Here to the End Product
tags:
  - big-idea
  - identity
  - research/plan
updated: 2026-09-19
---

# Roadmap

The arc from what exists today to the finished system, with the decision points that could redirect or end it. [[big_idea/09 Experiment Plan|Note 09]] describes the *studies*; [[big_idea/11 System Design|note 11]] describes the *target architecture*. This note is the sequence that connects them.

## Summary

Twelve stages, of which two are done. The critical path is short — **stages 2, 3 and 4** — because stage 4 produces the first number that says whether any of this works. Everything after it is currently an assumption. One stage (the fingerprint collector) must start early regardless, because it is gated on four weeks of calendar time rather than on effort.

## Where we are — 2026-09-19

| | Status |
|---|---|
| Browser keystroke capture, 13 session metrics, SQLite collector | **built**, 19 unit tests passing |
| Timing verified against kernel timestamps | **built and measured** — see [[big_idea/12 Measurements]] |
| Data collected | 5 sessions, ~750 keystrokes, **one subject** |
| Identity features | **none** |

> [!warning] The honest position
> The 13 metrics are **session-level averages, and averages cannot identify anyone**. A mean dwell of 189 ms is one number thousands of people share. The plumbing is built and verified; the features are not built at all. Nothing in this project has yet distinguished one person from another.

## The destination

**A score that decides when to demand stronger proof** — not a system that identifies people outright. Six vectors, scored per axis, fused as log-likelihood ratios, driving friction rather than denial. Full architecture in [[big_idea/11 System Design]].

The framing matters for the roadmap: we are never trying to reach "this is definitely Nishanth". We are trying to reach "this is unlike Nishanth enough to ask for a passkey".

## Stages

| # | Stage | Unlocks | Depends on | Effort | Status |
|---|---|---|---|---|---|
| 0 | Browser capture harness | Trustworthy raw event log | — | — | ✅ |
| 1 | Kernel timing verification | Confidence the clock is not lying | 0 | — | ✅ |
| **2** | **Per-key dwell + per-digraph flight** | The first real feature vector | 0 | ~1 day | **next** |
| **3** | **Fixed-passphrase mode** (`.tie5Roanl`) | Directly comparable samples | 2 | ~1 day | |
| **4** | **Second subject types it** | **The first EER** ★ **Gate A** | 3 | ~2 h + a friend | |
| 5 | Free challenges: A6 OTP timing, A1 perturbation, A2 chunking | Identity signal at zero friction | 3 | ~3 days | |
| 6 | Visible challenges: A5 scrambled keypad, A4, A8a | The cold-start answer ★ **Gate B** | 5 | ~4 days | |
| 7 | **Device fingerprint collector** | The device axis | 0 | ~2 days build, **4 weeks running** | **start in parallel** |
| 8 | Pointer capture and features | Second behavioural channel | 0 | ~3 days | |
| 9 | Scoring, calibration, fusion, decision bands | Signals become a decision | 4, 7 | ~1 week | |
| 10 | **The 2×2 study** | The research contribution ★ **Gate C** | 9 | ~2 weeks | |
| 11 | Write-up | | 10 | ~1 week | |

## Critical path

```
  2 ──► 3 ──► 4 ★ Gate A ──► 9 ──► 10 ★ Gate C ──► 11
                              ▲
  7 (start NOW, 4-week wall clock) ─────┘
```

**Stage 4 is where guesswork ends.** Two people typing the same passphrase, trained on early sessions and tested on later ones, yields an EER. Until that number exists, every dimension count in note 11 is a guess, and building stages 5–10 first risks elaborating a system on a false premise.

Stage 7 is the only thing that should run **in parallel from day one**, because drift and cross-session data cannot be manufactured later. Build the collector, get people visiting it weekly, and analyse in four weeks' time.

## Decision gates

> [!important] Gate A — after stage 4. Does keystroke timing separate people at all?
> Measure: **cross-session EER** on the fixed passphrase, ≥2 subjects, trained on early sessions and tested on later ones.
> - **< 10%** — the password vector is viable. Proceed as designed.
> - **10–20%** — marginal. More enrolment samples, or add the A2 chunk features, before committing.
> - **> 20%** — the password vector cannot carry identity alone. **Shift weight to the challenge vector and the device axis**, and reduce note 11's 49 password dimensions accordingly.

> [!important] Gate B — after stage 6. Is visible friction worth it?
> Measure: **identity information per second of user friction**, challenge versus passive observation over the same duration.
> - A 5-second challenge should beat roughly a minute of passive observation. If it does not, **drop the visible challenges (A4, A5, A8a) and keep only the free ones** (A6, A1) — the friction is not buying anything.
> - Also test A5's replay resistance explicitly: captured timings from a previous layout must not score.

> [!important] Gate C — after stage 10. Does the whole thesis hold?
> Measure: **behavioural EER conditioned on an identical device fingerprint** — the stranger at the owner's own unlocked laptop.
> - This is the number the project exists to produce, and almost nobody publishes it.
> - If it is not meaningfully better than chance, the honest conclusion is that **browser-only signals cannot defend the hard quadrant**, and the contribution is the negative result plus the measurement methodology. That is still a real contribution.

## What would make this stop

Written down in advance, so the decision is not made retrospectively:

- Gate A comes back above ~30% cross-session EER with no path to improvement.
- Gate C shows no separation, **and** the challenge vector also fails Gate B — meaning neither passive nor active behaviour works in a browser.
- Subject recruitment cannot reach the multi-device, multi-person design that [[big_idea/09 Experiment Plan|Experiment 3]] requires, making the central question unanswerable at any quality.

## Mapping to the experiment plan

| Stage | Experiment in [[big_idea/09 Experiment Plan]] |
|---|---|
| 2, 3, 4 | Experiment 2 — fixed-text keystroke baseline |
| 7 | Experiment 1 — fingerprint uniqueness and drift |
| 8 | Experiment 4 — free-text and detection delay |
| 5, 6 | Experiment 5 — active challenges and red team |
| 10 | Experiment 3 — the 2×2 ★ |

> [!note] Note 09's week-by-week plan predates the build
> It assumed the fingerprint collector would be written first. In practice the keystroke harness was built first and its timing verified against the kernel, which was not in the original plan at all and turned out to be worth doing. **This note supersedes note 09's "Build order" section**; the experiment designs and evaluation protocols in note 09 stand unchanged.

## Related notes

- [[big_idea/11 System Design|System design]] — the target architecture
- [[big_idea/09 Experiment Plan|Experiment plan]] — the studies behind each stage
- [[big_idea/12 Measurements|Measurements]] — what has actually been measured
- [[big_idea/00 Home|Big idea hub]]
