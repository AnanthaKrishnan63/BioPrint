---
title: BioPrint Hackathon — 36-Hour Build Plan
tags:
  - big-idea
  - identity
  - research/plan
  - hackathon
updated: 2026-09-19
---

# BioPrint Hackathon

How the research in this vault gets turned into a submission for **BioPrint: Behavior-Based Login Security** (Event 2 of the ROOT 36 hackathon, 36 hours). The source brief is `ROOT 36 Problem Statements.pdf` at the repo root. This note is the working plan for the build; [[big_idea/13 Roadmap|note 13]] is still the long-term research plan.

## Summary

BioPrint asks for a website login that **blocks** a user who has the correct password but the wrong behaviour, and that flags bots, with **no OTP or secondary-verification fallback**. Roadmap stages 2–4 (per-key features, a fixed passphrase, the first EER) are almost exactly the core of the brief. The CMU keystroke dataset gives a real cross-session EER over 51 subjects without recruiting anyone. Team of one working with parallel coding agents. Website, not a Chrome extension. Clock started around 2026-09-19 13:00, so the **deadline is about 2026-09-21 00:00 local**.

> [!important] What the demo must show
> (a) A genuine user enrolls and logs in successfully. (b) A teammate with the **correct password but the wrong behaviour**, or a scripted/bot attempt, is detected and **blocked**.

## The brief, distilled

| Item | Requirement | What we build | Link |
|---|---|---|---|
| Enrollment | Team-designed flow that captures a behavioural baseline | Type the password ~10 times; per-user profile is fitted on the 10th sample | [[big_idea/09 Experiment Plan\|Exp. 2 protocol]] |
| Fingerprinting engine | Models each user, **independent of the password** | Timing vector (H/DD/UD) + pointer features; content is only used for alignment | [[big_idea/11 System Design]] |
| Live check | Compare against the profile and **block on mismatch, no OTP fallback** | `POST /api/login` → allow/block + reasons | — |
| Bot detection | Scripts, bots and **replayed input** as a *distinct* fraud signal | Automation probe + implausible-timing rules + replay near-match | [[big_idea/07 Browser Signal Catalogue]] |
| Demo page | Genuine accepted, mimic/bot blocked | Login + enroll + dashboard pages | — |
| *Stretch:* adaptive profiles | Slowly follow natural drift | Fold accepted logins that sit well inside the threshold back into the profile | [[big_idea/11 System Design]] |
| *Stretch:* confidence dashboard | Why an attempt was accepted or flagged | Score vs threshold, top deviating features | — |
| *Stretch:* multi-modality | Touchpad / mouse / touch in one profile | Pointer channel (Agent C) | [[big_idea/07 Browser Signal Catalogue]] |
| *Stretch:* explainability | Human-readable reason for a block | Per-feature contributions → plain-English sentence | — |

### Judging weights and deliverables

| Criterion | Points | Where we earn it |
|---|---|---|
| Reliability | **25** | CMU cross-session EER + live tuning; FRR at fixed low FAR in the report |
| Overall innovation | 20 | Separate bot/replay signal, explainable decisions, a timing clock verified against the kernel ([[big_idea/12 Measurements]]) |
| Novelty in algorithms | 15 | Scaled Manhattan baseline (Killourhy & Maxion's best), per-user thresholds, fusion that keeps the signals separate |
| Creativity, ease of use, customer satisfaction, latency | 10 each | Pointer signals; a short enrollment; scoring in under 1 ms |

| Deliverable | Penalty if missing |
|---|---|
| Git repo with **real commit history** + README | Disqualification |
| Runnable website (localhost setup instructions) | −20 |
| 1–2 page report on how the algorithm works | −10 |
| Dummy login page + setup instructions | −20 |

## Mapping onto the roadmap

| Roadmap stage | In the hackathon | Why |
|---|---|---|
| 0–1 capture + kernel check | **Reused as is** | Already built and verified |
| 2 per-key dwell / digraph flight | **Core** (Agent A) | The first identity feature vector |
| 3 fixed-passphrase mode | **Core**: the password field *is* the fixed passphrase | Directly comparable samples |
| 4 first EER, Gate A | **Core**, done on CMU data plus one live impostor | Reliability is the largest criterion |
| 5–6 challenges | Out of scope | Friction the brief does not reward |
| 7 device fingerprint collector | **Reduced to a 2–3 h automation probe** feeding bot detection | The 4-week drift study can't fit in 36 h; the demo is the same laptop, so the device axis carries no information there |
| 8 pointer capture | **In** (Agent C) | Second behavioural channel, stretch goal |
| 9 scoring / fusion | **In**, simplified: one `decide()` function | Needed to block anyone |
| 10 2×2 study | **Out of scope** | Future work in the report |

## Conventions this overrides

> [!warning] Hackathon-only trade-offs, never to be carried back into real work
> | Research convention | Hackathon rule | Reason |
> |---|---|---|
> | Respond with *friction*, not denial; passkeys short-circuit | **Block** on mismatch | The brief explicitly forbids OTP or secondary-verification fallback |
> | Two orthogonal axes, device + behaviour | Behaviour + **bot** signal; the device axis becomes an automation probe | The demo impostor is on the owner's own laptop, where the fingerprint matches perfectly |
> | `event.code` only, *so it isn't a keylogger* | `event.code` **is logged on the password field**, which on QWERTY spells the password; `bioprint.db` stores it | Deliberate, user-approved choice: simpler server-side pairing, and exposing passwords in a local hackathon DB is acceptable |
> | Report FRR at fixed low FAR | **Still applies** to the report | The honest metric doesn't change because the product blocks |

## CMU dataset: Gate A without recruiting

| Property | Value | Link |
|---|---|---|
| Source | Killourhy & Maxion, DSN 2009, `DSL-StrongPasswordData.csv` (~4.7 MB, not committed) | [[big_idea/05 Datasets and Data Availability]] |
| Scale | 51 subjects × 400 repetitions of `.tie5Roanl`, 8 sessions on separate days | [[big_idea/02 Papers]] |
| Layout | Per keystroke **H** (hold); per consecutive pair **DD** (keydown→keydown) and **UD** (keyup→keydown), in seconds | — |
| Protocol | Train on sessions 1–4, test on 5–8 → honest **cross-session** EER / FRR@FAR | [[big_idea/13 Roadmap\|Gate A]] |

Our live feature vector uses **the same H/DD/UD layout (in ms)**, so one `fit()` / `score()` implementation runs on both CMU data and live logins.

## Phases

| Phase | Work | Output |
|---|---|---|
| **0** (main session, ~1 h) | Contracts (`contracts.py`), DB (`db.py`), API (`server.py`), **stubs for every engine function** so the app runs end to end on day one; numpy + pytest added; `*.csv` ignored | A running skeleton that agents fill in |
| **1** (five parallel agents) | See the ownership table below | Real engine, bot detector, pointer channel, UI, CMU EER |
| **2** (~8 h) | Integration; Gate A on CMU; threshold tuning; **Playwright bot + replay attack** as live demo villains; adaptive profile if green | Demo-ready system |
| **3** (~6 h) | README, 1–2 page report, demo rehearsal, ~4 h buffer | Submission |

### Phase 1 file ownership

| Agent | Owns | Delivers |
|---|---|---|
| **A: Engine** | `engine/features.py`, `engine/scorer.py` | H/DD/UD features, scaled Manhattan `fit`/`score`, per-user threshold by scoring each enrollment sample against the rest |
| **B: Bot** | `engine/bot.py`, `static/probe.js` | `webdriver`, headless UA, SwiftShader/llvmpipe renderer, `isTrusted`; timings too regular, paste, replay near-match, no pointer movement |
| **C: Pointer** | `engine/pointer.py`, `static/pointer.js` | Path to the Login button, hesitation before clicking, click hold time |
| **D: UI** | `static/*.html`, `static/app.js` | Login, enroll, dashboard with plain-English reasons |
| **E: Eval** | `eval/cmu.py` | CMU loader + cross-session EER / FRR@FAR harness calling Agent A's scorer |

Every signal returns the same shape, `{score, threshold, contributions, reasons[]}`. **`decide()` keeps them separate** and never collapses them to one number: a bot flag blocks, a keystroke mismatch blocks, pointer data adds weight. The exact rule gets tuned in phase 2.

## API and storage

| Endpoint | Does |
|---|---|
| `POST /api/register` | username + password (scrypt hash) |
| `POST /api/enroll` | one sample; returns progress (x / 10); fits the profile on the 10th |
| `POST /api/login` | password + sample → `{decision, reasons, scores, bot_flags, latency_ms, attempt_id}` |
| `GET /api/attempts?user=` | data for the dashboard |

A sample is `{keystrokes: [{code, type, t, trusted}], pointer, env, meta: {had_paste, corrections}}`. The server pairs keydown/keyup itself and must tolerate Firefox's phantom duplicate keydowns ([[big_idea/12 Measurements]]).

| Table (`bioprint.db`, separate from `sessions.db`) | Holds |
|---|---|
| `users` | username, password hash, enrolled flag |
| `samples` | kind (enroll/login), raw key events, pointer, env, features |
| `attempts` | decision, scores, bot flags, latency |

## Human dependencies

> [!important] Only the person can do these
> 1. **One real person for ~20 minutes** to be the live impostor with the correct password. CMU data can't replace that in a live demo.
> 2. **Deadline ≈ 2026-09-21 00:00 local.** Hours 0–12 are the must-haves; after that, cut from the bottom of the phase list.

## Related notes

- [[big_idea/13 Roadmap|Roadmap]]: the long-term research plan this note borrows from
- [[big_idea/11 System Design|System design]]: the target architecture this simplifies
- [[big_idea/09 Experiment Plan|Experiment plan]]: Experiment 2 is the fixed-text protocol used here
- [[big_idea/12 Measurements|Measurements]]: timing accuracy and phantom keydowns
- [[big_idea/05 Datasets and Data Availability|Datasets]]: the CMU dataset
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]: automation-probe signals
- [[big_idea/00 Home|Big idea hub]]
