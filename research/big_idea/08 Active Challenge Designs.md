---
title: Active Challenge Designs
tags:
  - big-idea
  - identity
  - research/design
  - active-challenge
updated: 2026-09-18
---

# Active challenge designs

Developed from [[yooooooooo|the original brainstorm]]. Where [[big_idea/07 Browser Signal Catalogue|Layer D]] observes the user *passively* and slowly, an active challenge **provokes** a measurable response in seconds.

## Why active beats passive here

| Property | Passive behavioural | Active challenge |
|---|---|---|
| Time to a usable decision | 300–1000 keystrokes, or minutes of pointer activity | **2–8 seconds** |
| Signal-to-noise | Low — the user is doing an arbitrary task | **High — the stimulus is controlled and identical every time** |
| Cross-session comparability | Poor; behaviour depends on the task and page | **Excellent; the probe is fixed, so samples are directly comparable** |
| Replay resistance | Weak — captured timings replay cleanly | **Strong if the probe is randomised per session** |
| Cold start | 5–30 sessions | **1–3 sessions**, because the measurement is controlled |
| User friction | Zero | Non-zero — the central cost |

> [!important] The core insight
> Passive behavioural biometrics has poor SNR because it measures an *uncontrolled* stimulus. An active challenge is a **controlled psychophysical experiment run in the browser**: identical stimulus, measured response, comparable across sessions. That is why it converges in seconds rather than minutes.

> [!warning] Prior art
> **US10298614B2 (BioCatch)** already claims intentionally-introduced input/output interference to elicit user reactions from corrective pointer movement — i.e. design A1 below. Research and coursework are unaffected; commercial use needs a freedom-to-operate opinion. See [[big_idea/04 Patents]].

---

## A1 — Perturbation probe ("trick the user")

*From brainstorm item 1.*

**Mechanism.** Introduce a small, brief, deliberate anomaly and measure the correction. Examples: displace the cursor by 15–30 px for 200 ms; delay a button's visual response by 150 ms; shift a form field 10 px as the user reaches for it; swap the position of two buttons momentarily; drop one keystroke's echo.

**What it measures.** Not motor style but **sensorimotor feedback control** — an involuntary closed-loop response that is much harder to consciously mask than typing rhythm.

| Feature | Definition |
|---|---|
| Reaction latency | Perturbation onset → first corrective movement |
| Correction gain | Overshoot/undershoot ratio in the corrective movement |
| Correction profile | Single ballistic correction vs. multiple small adjustments |
| Settling time | Onset → stable re-acquisition of the target |
| Noticing rate | Does the user pause, and for how long? |
| Modality of response | Mouse correction vs. keyboard vs. abandon-and-restart |

**Friction:** very low if subtle — most users do not consciously register a 200 ms displacement.
**Risk:** accessibility. Users with motor impairments, tremor, or using switch/eye-tracking input will respond very differently and may be harmed by the perturbation itself. **Must be disableable, and must never perturb a destructive control.**

---

## A2 — Password motor-chunk structure

*From brainstorm item 2: "clustering of letters — any possible combinations".*

**Mechanism.** This is the sharpest idea in the brainstorm. A practised password is not typed as *n* independent keystrokes: it is typed as a small number of **motor chunks** learned as units. Chunk boundaries appear as **local maxima in the digraph-latency profile**, and the chunking pattern is a property of *how that person learned that password* — not of the password itself.

**Why it matters for security.** An attacker who *knows* the password still types it as a novice: uniformly slow, no chunk structure, boundaries in the wrong places. Knowledge of the secret does not confer the motor program.

| Feature | Definition |
|---|---|
| Chunk segmentation | Segment the digraph-latency sequence at latency peaks → chunk boundary vector |
| Chunk count and sizes | e.g. `[4, 3, 3]` for a 10-character password |
| Within-chunk vs. between-chunk latency ratio | The clearest single discriminator; a novice's ratio ≈ 1 |
| Chunk-internal variance | Practised chunks are low-variance |
| Boundary stability across repetitions | Genuine users segment identically every time |
| Rollover frequency within chunks | Negative flight times cluster inside chunks |

**Combinatorial framing** (the "any possible combinations" part): for an *n*-character password there are 2^(n−1) possible boundary placements. The observed boundary vector is therefore a discrete code of up to n−1 bits **on top of** the continuous timing features — and it is far more quantisation-robust than raw dwell times, which matters given timer coarsening.

**Friction:** zero. The user is already typing their password.
**This is the highest value-per-unit-friction design in the whole set.**

---

## A3 — Frequency decomposition of the interaction signal

*From brainstorm item 3.*

**Mechanism.** Treat the interaction stream as a **time series** and work in the frequency domain rather than with hand-crafted summary statistics.

| Stream | Construction | Spectral features |
|---|---|---|
| Typing | Keystroke onsets → inter-key interval series, or an impulse train resampled to fixed rate | Dominant typing frequency, harmonic structure, spectral centroid, spectral flatness, band energy ratios |
| Pointer | Velocity magnitude resampled to fixed Hz | **Tremor band (8–12 Hz)** power, sub-movement rate (~2–5 Hz), spectral rolloff |
| Scroll | Wheel delta series | Flick periodicity, momentum decay constant |
| Correction | Backspace-event impulse train | Burst periodicity |

**Why it is a good idea.** Physiological tremor sits in a narrow 8–12 Hz band and is **individually characteristic and largely involuntary** — you cannot decide to change it. Spectral features are also naturally invariant to the absolute speed at which someone is working, which is one of the biggest sources of within-user variance in time-domain features.

**Methods:** Welch PSD, or wavelet decomposition for non-stationary signals. Recurrence plots plus a vision model (Mazumdar & Sundaram 2025) are the deep-learning analogue — see [[big_idea/02 Papers]].

**Caveat:** needs adequate sampling. Use `getCoalescedEvents()` for pointer; keystroke event rates are too low for meaningful spectra above a few Hz, so apply spectral analysis mainly to pointer and scroll.

---

## A4 — Cognitive timing via arithmetic challenge

*From brainstorm item 4: "mental math speed captured from captcha".*

**Mechanism.** Present a small arithmetic problem (e.g. `7 × 8 + 13`) as a lightweight captcha. Measure not just correctness but the **full response timeline**.

| Feature | Definition |
|---|---|
| Total response latency | Display → first keystroke |
| Latency **scaling with difficulty** | Regress latency on problem difficulty; the **slope** is the personal parameter, not the intercept |
| Answer-entry rhythm | Digit-by-digit typing pattern of the answer |
| Hesitation position | Where within the answer the user pauses (indicates carrying/decomposition strategy) |
| Self-correction | Backspaces during answer entry |
| Strategy signature | Which problem *types* are fast for this person — a stable cognitive profile |

**Why it works.** Arithmetic fluency is stable within a person and highly variable between people, and the *latency-versus-difficulty slope* is far more personal than raw speed. It doubles as a bot/LLM check: an automated solver is either implausibly fast or implausibly uniform.

**Friction:** moderate and visible. Best reserved for step-up, not every login.
**Risk:** dyscalculia, numeracy differences, and non-native numeral familiarity make this **inequitable as a hard gate**. Use as one score among several and always offer an alternative.

---

## A5 — Scrambled keypad entry

*From brainstorm item 5. The strongest security property in the set.*

**Mechanism.** Present an on-screen keypad (or keyboard) with a **randomised layout**, different every session, and have the user enter a PIN or password by clicking.

**Why this is genuinely clever.** It inverts what is being measured. A normal keyboard measures *motor memory*, which an observer can learn to imitate. A scrambled layout destroys motor memory and forces **visual search**, so what gets measured is the user's **visual scanning and target-acquisition behaviour** — a different and less imitable trait.

Three properties fall out of the randomisation:

1. **Replay-proof.** Captured timings from a previous session are meaningless against a new random layout. This is a real security guarantee that passive behavioural biometrics cannot offer.
2. **Shoulder-surf resistant.** Observed click positions do not reveal the secret.
3. **Difficulty is controllable.** Scramble distance from a standard layout can be dialled up or down as an experimental variable.

| Feature | Definition |
|---|---|
| Per-digit search time | Previous click → next click, normalised by on-screen distance |
| **Fitts-normalised movement time** | Residual after removing the distance/width effect = the personal component |
| Search strategy | Reading-order scan vs. spatial-memory jumps; inferred from the mouse path between clicks |
| Learning curve within a session | How quickly the user memorises the new layout across digits — a **working-memory signature** |
| Path efficiency | Actual path length ÷ straight-line distance per digit |
| Error rate and correction | Mis-clicks and how they are fixed |

**Friction:** high and very visible. Reserve for step-up or high-value actions.
**Bonus:** on mobile, the on-screen keypad gives you `Touch.force` and contact radius — signals a hardware keyboard cannot provide.

---

## A6 — Two-factor code entry timing

*From brainstorm item 6. Zero added friction — it instruments an existing step.*

**Mechanism.** When an OTP is already required, instrument the whole interaction rather than just checking the code.

| Feature | Definition | What it reveals |
|---|---|---|
| Code-request → first-keystroke latency | Time to retrieve the code | Device-switching habit, physical setup |
| Entry mode | Autofill (OS suggestion) vs. paste vs. typed | Owners on their own device usually autofill; attackers relaying a code type it |
| Digit-group rhythm | Pauses within a 6-digit code | Reveals chunking: `123-456` vs. `12-34-56` — personal and stable |
| `visibilitychange` / `blur` pattern | Tab or window switches during entry | Phone-in-hand vs. reading from another window vs. relayed by a third party |
| Retry and expiry behaviour | Does the code expire before entry? | Habitual pace |
| Total elapsed time distribution | | Stable per user and per setup |

**High-value fraud signal.** In a **social-engineering / OTP-relay attack**, the victim reads the code aloud to the attacker, who types it. That produces a distinctive pattern: long request-to-entry latency, typed rather than autofilled, no `blur` to a phone app, and unusual digit rhythm. BioCatch's commercial product targets precisely this class of fraud (see [[big_idea/03 Startups and Products]]).

---

## A7 — Device-conditioned behaviour and per-device normalisation

*From brainstorm items 3 and 7: "typing speed (different devices)".*

**The observation.** Typing speed, dwell times and pointer dynamics are **not properties of the person alone** — they are properties of the *person on a particular device*. A mechanical keyboard, a laptop chiclet keyboard and a phone touchscreen produce three different distributions for the same human. A trackpad and a mouse produce two different pointer signatures.

This is usually treated as a nuisance. It is actually **two useful things at once**:

**(a) A normalisation requirement.** Behavioural templates must be stored **per (user, device-class)** pair, not per user. Comparing a session on Device B against a template built on Device A will false-reject almost every time. This alone explains a large share of the gap between published lab EERs and production performance.

| Device class | Keyed on | Model |
|---|---|---|
| Desktop + external mechanical keyboard | Fingerprint cluster + `pointerType` + wheel delta quantum | Full keystroke + pointer model |
| Laptop internal keyboard + trackpad | Same, with trackpad-specific scroll signature | Separate template |
| Phone / tablet touchscreen | `pointerType: touch`, screen size | Motion + touch model; **keystroke timing largely useless** |

**(b) A signal in its own right.** The *relationship between* a user's device profiles is personal: how much faster they are on their desktop than their laptop, whether their pointer tremor differs between mouse and trackpad. A user with an established Device A template who appears on a new Device B can be partly verified by checking whether the **shift** matches the shift that user showed historically when moving between their own devices.

**Cold-start consequence.** A brand-new device class has no template. This is exactly the top-right quadrant of the [[big_idea/00 Home|2×2]] — and the reason Layer A cryptographic binding matters: a passkey transfers assurance across devices when behaviour cannot.

**Feature set.**

| Feature | Definition |
|---|---|
| Device-class label | Derived from fingerprint cluster + `pointerType` + input-event characteristics |
| Per-class template | Separate μ, σ per (user, device class) |
| Cross-device ratio vector | e.g. `median_dwell(B) / median_dwell(A)` — stable per user |
| Wheel delta quantum | Hardware-level: mouse wheels emit fixed deltas, trackpads emit continuous ones. **Cleanly separates trackpad from mouse** |
| Key-repeat rate | OS setting, stable per device, leaks on held keys |

---

## A8 — Visual reaction probe ("flashing lights")

*From brainstorm item 8.*

Two distinct things can be built from a flashing visual stimulus, and they belong to different layers.

### A8a — Reaction-time psychophysics (Layer D)

**Mechanism.** Flash a visual stimulus at a randomised delay and measure the response. This is a **simple reaction-time test**, one of the oldest and best-characterised measurements in experimental psychology.

| Feature | Definition | Notes |
|---|---|---|
| Simple reaction time | Stimulus onset → response | Mean ~200–250 ms; **within-person variance is smaller than between-person variance** |
| RT **distribution shape** | Fit an ex-Gaussian: μ, σ, τ | The τ (exponential tail) parameter is the personal one; mean RT alone is weak |
| Choice reaction time | Respond differently by stimulus colour/position | Adds a decision stage; scales with alternatives (Hick's law) — the **slope** is personal |
| Anticipation rate | Responses before stimulus onset | Strategy signature |
| Vigilance decrement | RT drift across repeated trials | Attention profile |

**Strengths.** Fast (10 trials ≈ 10 seconds), controlled, robust to timer coarsening (RTs are hundreds of ms, so 1 ms quantisation is irrelevant — unlike dwell times). Also a strong **liveness/bot check**: automated responses are either sub-human-fast or implausibly low-variance.

**Weaknesses.** RT is heavily state-dependent — fatigue, caffeine, time of day, distraction all shift it by tens of milliseconds. Use **distribution shape**, not mean, and expect a wide acceptance band.

> [!warning] Photosensitive epilepsy — a hard constraint
> Flashing visual stimuli can trigger seizures. **WCAG 2.1 SC 2.3.1 prohibits more than three flashes per second**, and SC 2.3.2 recommends none at all. Any implementation must stay under three flashes/second, avoid saturated red, keep the flashing area small, respect `prefers-reduced-motion`, and warn beforehand. A single non-repeating stimulus change — a colour shift or a shape appearing — gives the same reaction-time measurement with none of the risk. **Prefer that; do not literally flash.**

### A8b — Display refresh-rate and rendering fingerprint (Layer B)

The same mechanism, repurposed as a *device* signal rather than a person signal. Measuring `requestAnimationFrame` intervals reveals the **display refresh rate** — 60 / 90 / 120 / 144 / 165 Hz — plus frame-timing jitter characteristic of the GPU and compositor.

| Feature | Notes |
|---|---|
| Refresh rate | 2–4 bits of entropy; very stable per device |
| Frame-time jitter distribution | GPU/compositor signature |
| Dropped-frame behaviour under synthetic load | Coarse GPU performance tier |

This belongs in [[big_idea/07 Browser Signal Catalogue|the signal catalogue]] rather than here — it identifies the machine, not the person. It is a genuinely useful addition to the fingerprint because a laptop's internal 120 Hz panel versus an external 60 Hz monitor is a stable, hard-to-spoof distinction.

---

## Deployment policy

Challenges cost friction, so trigger them by risk rather than always.

| Situation | Response |
|---|---|
| Valid passkey / DBSC binding | **No challenge.** Cryptographic proof outranks everything. |
| Known device, familiar context, sufficient passive behavioural history | No challenge |
| Known device, **unfamiliar behaviour** (the hard quadrant) | **A1 perturbation** + **A2 password chunk analysis** — both invisible |
| New device, familiar behaviour | Low-friction confirmation; **A6** if an OTP is already in flow |
| New device, unfamiliar behaviour | **A5 scrambled keypad** or full step-up to passkey |
| Sensitive action (payment, password change, data export) | **A5** regardless of prior score |
| Suspected bot or RAT | **A4** (cognitive) — automation fails the difficulty-scaling relationship |

> [!important] Ranking by value per unit of friction
> 1. **A2 password chunk structure** — zero friction, high signal, quantisation-robust, no prior-art collision. **Build this first.**
> 2. **A6 OTP timing** — zero added friction, instruments an existing step, catches OTP-relay fraud.
> 3. **A3 frequency decomposition** — zero friction, applies to passively collected pointer data, speed-invariant.
> 4. **A1 perturbation** — very low friction, strong involuntary signal, but patent-encumbered and accessibility-sensitive.
> 5. **A5 scrambled keypad** — high friction, but uniquely **replay-proof**; the right tool for step-up.
> 6. **A8a visual reaction probe** — fast, timer-coarsening-immune, good bot check, but state-dependent and subject to a hard photosensitivity constraint.
> 7. **A4 arithmetic** — visible friction and equity concerns; use as a bot check rather than an identity check.
>
> **A7 is not a challenge — it is a correctness requirement.** Per-device normalisation must be in place before any behavioural number is trustworthy.

> [!important] Where these land in the system
> The challenges become **vector 4 of six** in [[big_idea/11 System Design]], ~46 dimensions, with an enrolment schedule that front-loads them at signup. That placement matters: they are the only vector whose sampling rate is controlled rather than observed, so they mature first and are what narrows the cold-start gap from ten logins to about five.

## Related notes

- [[big_idea/11 System Design|System design]] — where these fit in the whole
- [[yooooooooo|Original brainstorm]]
- [[big_idea/04 Patents|Patents]] — prior art on A1
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
