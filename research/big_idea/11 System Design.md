---
title: System Design — Signup to Login
tags:
  - big-idea
  - identity
  - research/design
updated: 2026-09-19
---

# System design: what is collected, and how a login is judged

The first end-to-end architecture for the system the rest of this folder argues for. Read [[big_idea/07 Browser Signal Catalogue|the signal catalogue]] for what each signal is, and [[big_idea/06 Frameworks and Validation|Frameworks and validation]] for how it is evaluated. This note connects them: what gets stored at signup, and what happens on every login afterwards.

## Summary

Roughly **340 numbers per user, per device class**, across six vectors. The shape of the system is set by one constraint: **a signup session cannot produce a passive behavioural template**. Signup yields 60–100 keystrokes from one sitting on one device, and no honest evaluation protocol accepts a single-session template. Enrolment is therefore **progressive** — device and context work immediately, passive behaviour matures over 10–30 logins.

The **challenge-response vector is what narrows that gap**. A controlled probe does not wait for repetition, and signup is precisely where a user tolerates friction, so three or four challenges at signup buy a usable baseline by around the fifth login rather than the tenth.

> [!important] The constraint that shapes everything
> **At the first login after signup, passive behaviour cannot protect the account.** This is inherent, not a gap to engineer around. Any design claiming passive behavioural verification from one prior session is either overfitting to a single sitting or lying about its cold start.
>
> Challenges narrow the gap but do not close it — three samples give a usable mean and a poor variance estimate, so the challenge vector is weak until roughly the fifth login. **A passkey enrolled at signup remains the only thing giving real assurance on day one.**

---

## The six vectors

Stored per user **and per device class** — a desktop template cannot score a phone session, because typing speed and pointer dynamics are properties of *the person on a particular device*, not of the person alone (see [[big_idea/08 Active Challenge Designs|A7]]).

### 1. Password timing vector — strongest, and free

The one piece of fixed text typed on every single login. For an *n*-character password:

| Component | Dims at n=12 |
|---|---|
| Dwell per position | 12 |
| Flight per position (up → down) | 11 |
| Down-down latency per position | 11 |
| Chunk boundary vector (binary; see [[big_idea/08 Active Challenge Designs\|A2]]) | 11 |
| Total entry time, shift/capslock strategy | ~4 |
| **Total** | **~49** |

This is the CMU protocol and it costs the user nothing — they are already typing it.

> [!note] Two properties worth stating explicitly
> Timings are stored **per position**, never per character, so the template reveals nothing about the password itself.
> A **password change invalidates this vector entirely**. The other four survive, and the system falls back to them while this one re-enrols.

### 2. Free-text vector — typing anywhere in the app

| Component | Dims |
|---|---|
| Per-key dwell: mean + sd for ~30 common keys | 60 |
| Per-digraph flight: mean + sd for top ~50 pairs | 100 |
| Overlap degree distribution (how often 2+ keys are held at once) | ~5 |
| Rollover rate, correction rate, correction latency, burst structure | ~8 |
| **Total** | **~173** |

Necessarily **sparse at first** — there is no `KeyZ` dwell until the user types a Z — and fills in across sessions. This is the vector that makes *continuous* checking possible rather than login-only.

### 3. Pointer vector — ~25 dims

Fitts's law `(a, b)`, velocity profile percentiles, path curvature, tremor spectrum (8–12 Hz), click dwell, double-click interval, scroll signature.

### 4. Challenge-response vector — ~46 dims

The four vectors above are **passive**: collected for free, continuously, from whatever the user happens to do. This one is **active** — it comes from a deliberate probe, and it therefore behaves differently in every respect that matters. Designs are in [[big_idea/08 Active Challenge Designs]].

| Challenge | Features | Dims | Friction |
|---|---|---|---|
| **A6 OTP entry timing** | Request → first-keystroke latency, entry mode (autofill / paste / typed), digit-group rhythm, `blur` pattern during entry, total elapsed | ~8 | **none** — instruments a step already in the flow |
| **A1 Perturbation probe** | Reaction latency to the perturbation, correction gain (over/undershoot), correction profile (one ballistic move vs. several small ones), settling time, noticing rate | ~8 | **invisible** — most users never consciously register it |
| **A4 Cognitive / arithmetic** | Response latency, **latency-vs-difficulty slope**, answer-entry rhythm, hesitation position, self-correction, accuracy by problem type | ~10 | visible |
| **A5 Scrambled keypad** | Per-digit search time normalised by distance, **Fitts-normalised movement residual**, search strategy from the pointer path, within-session learning curve, path efficiency, misclick rate | ~12 | high |
| **A8a Visual reaction** | Simple RT as ex-Gaussian (μ, σ, **τ**), choice RT and Hick slope, anticipation rate, vigilance decrement | ~8 | visible |

> [!important] Two features here are the personal parameter; the raw speed is not
> For **A4**, it is the *slope* of latency against problem difficulty, not the intercept — how fast someone is at arithmetic overall varies with mood and caffeine, but how sharply they slow as problems get harder is stable. For **A8a**, it is **τ**, the exponential tail of the reaction-time distribution, not the mean. For **A5**, it is the *residual* after Fitts's law accounts for distance and target size. In each case the naive summary statistic is the weak feature and the shape parameter is the strong one.

### Why this vector changes the architecture

**It is the answer to the cold-start problem.** The passive vectors are unusable for roughly ten logins because they need repetition the user has not yet supplied. A challenge does not wait for repetition — it is a controlled stimulus with a controlled response, so samples are directly comparable from the very first one.

And the friction lands where it is cheapest. **At signup the user already expects setup effort**, so three or four challenges cost almost nothing in goodwill — whereas the same challenges on an ordinary login would be intolerable.

| | Passive vectors | Challenge vector |
|---|---|---|
| Samples needed to be usable | 10–30 sessions | **5–8 samples** |
| Signal-to-noise | low — uncontrolled stimulus | **high — identical stimulus every time** |
| Cross-session comparability | poor; depends on the task | **excellent; the probe is fixed** |
| Replay resistance | weak | **strong for A5** — the layout is randomised each time, so captured timings are worthless |
| Cost | free | friction, except A6 and A1 |

**A5's replay resistance has no equivalent among the passive vectors.** Randomising the keypad layout every session means an attacker who has recorded a previous session cannot replay it, which is the one attack passive behavioural biometrics simply cannot defend against.

### Enrolment and deployment schedule

Challenges cost friction, so they are not collected like the passive vectors.

| When | What runs | Why |
|---|---|---|
| **Signup** | 3 challenges, ~30–45 s total (A5 + A4 + A8a) | User tolerance for setup is at its highest; buys a usable baseline immediately |
| **Logins 2–5** | One challenge per login, rotating | Reaches 5–8 samples per challenge type |
| **Every login thereafter** | **A6** whenever an OTP is already in the flow; **A1** silently on any page | Both are free |
| **On risk** | A5 or A4, chosen by which axis is in doubt | Friction only when it has been earned |
| **Before sensitive actions** | A5 regardless of score | Payment, password change, data export |

> [!warning] Three constraints that are not optional
> **State dependence.** Reaction time and arithmetic speed shift by tens of milliseconds with fatigue, time of day and distraction — far more than typing rhythm does. Use distribution shape (τ, slope) rather than means, and accept wider bands.
> **Equity.** A4 is inequitable as a hard gate (dyscalculia, numeracy differences, non-native numerals). A8a carries a hard photosensitivity limit — WCAG 2.3.1 forbids more than three flashes per second, and a single non-repeating stimulus change gives the same measurement with none of the risk. Both need alternatives.
> **Prior art.** A1 is claimed by **US10298614B2 (BioCatch)**. Irrelevant for research; needs a freedom-to-operate opinion for a product. See [[big_idea/04 Patents]].

### 5. Device vector — ~40 attributes

Canvas hash, WebGL/WebGPU renderer, font probe bitmap, audio hash, screen + DPR, display refresh rate, timezone, languages, hardwareConcurrency, UA-CH — plus server-side JA4 and HTTP/2 fingerprint. Stored as **individual attributes, never hashed together**, and fuzzy-matched.

### 6. Context vector — ~8

IP, ASN, ISP class, geo, login hour, weekday, inter-login interval, entry path.

---

## What a login actually does

```
1. COLLECT      the same five vectors from this session
2. NORMALISE    z = (x - mu_user) / sigma_user
                per feature, against THIS user's own enrolled statistics
3. SCORE        each axis independently:
                  S_device   = weighted fuzzy match against known devices
                  S_context  = ASN familiarity, impossible travel, login hour
                  S_password = scaled Manhattan distance to the password template
                  S_freetext = same, over whichever keys actually appeared
                  S_pointer  = same
                  S_challenge= same, for whichever probes have run this session
4. CALIBRATE    each score -> LLR = log P(s | genuine) / P(s | impostor)
5. FUSE         sum the LLRs that are available; omit the ones that are not
6. DECIDE       band -> silent / monitor / step-up / block and notify
```

Three mechanics matter more than the choice of model:

**Normalisation is what makes it personal.** A 189 ms dwell means nothing in absolute terms. Against *this user's* enrolled 189 ± 25 ms, a 120 ms sample is −2.8σ, and that means something.

**Calibration needs other users.** You can measure that a sample sits 2.8σ from the template, but "how suspicious is 2.8σ?" can only be answered by knowing how often *other people* land there. That is the global impostor pool, and it is why step 4 is impossible with a single user in the database.

**Missing signals are skipped, not imputed.** This is the entire reason for per-axis LLRs rather than one monolithic model: a login page has no free text and no mouse path, so those terms contribute nothing rather than contributing noise. See [[big_idea/06 Frameworks and Validation|the fusion section]].

---

## Maturity timeline

| Login # | Available | Real protection |
|---|---|---|
| Signup | Device, context, passkey, **3 challenge samples** | **Device + passkey**; challenge baseline recorded but too thin to score |
| 2–5 | One challenge per login → 5–8 samples each | Device + context; **challenge vector becomes scoreable ~login 5** |
| ~10 | Password vector usable (~10 samples over ≥2 days) | + first *passive* behavioural signal |
| 15–30 | Free-text and pointer vectors usable | Full fusion |
| 30+ | Template adaptation running | Mature |

The ordering is the point: **the challenge vector matures first**, because it is the only one whose sampling rate you control rather than merely observe.

---

## Storage shape

```sql
user_devices   (user_id, device_id, first_seen, last_seen,
                attributes_json, n_sessions)

user_templates (user_id, device_class, feature, mean, sd, n_samples, updated_at)
               -- feature is a string key:
               --   "pw_dwell:3", "dwell:KeyA", "flight:KeyT>KeyH", "fitts_b",
               --   "a4_difficulty_slope", "a5_fitts_residual", "a8_rt_tau"
```

Long format rather than wide, for two reasons: features can be added without migrating a 300-column table, and **`n_samples` per feature** is what decides whether a given feature is yet mature enough to contribute to the score.

---

## Build order, and the honest caveat

> [!warning] This is architecture for an untested assumption
> Everything above presumes that keystroke timing separates people at all — which has not yet been tested on data from this project. The shortest path to knowing is [[big_idea/09 Experiment Plan|Experiment 2]]: one more person typing a fixed passphrase gives an EER, and that number decides whether the password vector deserves 49 dimensions or is a dead end. **Build stages 2–4 of the build log before building any of this.**

See [[big_idea/12 Measurements|Measurements]] for what has actually been measured so far.

## Related notes

- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]] — what each passive signal is
- [[big_idea/08 Active Challenge Designs|Active challenge designs]] — the probes behind vector 4
- [[big_idea/06 Frameworks and Validation|Frameworks and validation]] — scoring, thresholds, evaluation
- [[big_idea/09 Experiment Plan|Experiment plan]] — the studies that test this
- [[big_idea/12 Measurements|Measurements]] — own-machine results
