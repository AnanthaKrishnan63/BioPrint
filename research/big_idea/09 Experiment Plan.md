---
title: Experiment Plan
tags:
  - big-idea
  - identity
  - research/method
updated: 2026-09-18
---

# Experiment plan

Five studies, ordered by effort. Each is self-contained and produces a defensible number. Constraints are not yet fixed, so each states its own minimum viable scale.

> [!important] Experiment 3 is the point
> Experiments 1, 2, 4 and 5 replicate or extend known work. **Experiment 3 is the one nobody publishes and the one that answers the project's actual question.** If time is short, do 1, 2 and 3.

---

## Experiment 1 — Fingerprint uniqueness and the minimal feature set

**Question.** What is the smallest set of browser attributes that uniquely identifies a device *and* survives a month of drift?

**Setup.** A static page collecting ~40 attributes (fork FingerprintJS v3 and extend from [[big_idea/07 Browser Signal Catalogue|the catalogue]]), POSTing to a collector with a subject ID. **N ≥ 25 volunteers, each on ≥ 2 devices and ≥ 2 browsers**, revisiting **weekly for 4 weeks**. Capture the JA4 and HTTP/2 fingerprints server-side on every visit.

**Analysis.**
1. Per-attribute **Shannon entropy** `H = −Σ p log₂ p`, and normalised `H / log₂(N)`.
2. Per-attribute **stability**: `P(value unchanged | same device, Δt)` at 7, 30 days.
3. **Greedy forward selection** — repeatedly add the attribute maximising uniquely-identified devices, subject to `stability > 0.95`. Plot *k features → % devices uniquely identified*.
4. **Anonymity-set size** distribution.
5. JA4-vs-UA consistency: how often do they disagree, and on which browsers?

**Deliverable.** "With these *k* attributes you distinguish 100% of devices and the fingerprint survives 30 days." This is the "unique combination of features" the project set out to find.
**Effort:** ~1 week including the collection window. **The collection window is the cost, not the code.**

---

## Experiment 2 — Fixed-text keystroke baseline (the simple test)

**Question.** What EER is achievable from a single password entry, under an honest cross-session protocol?

**Setup.** **N ≥ 20 subjects** each type a fixed string — use `.tie5Roanl` so results are directly comparable to the CMU benchmark — **50 times across ≥ 4 sessions on different days**. Log `(code, event, t)` triples only. Record browser and OS per session.

**Features.** 10 dwell + 9 flight + 9 down-down = **28 dimensions**.
**Model.** Per-user **scaled Manhattan distance** to the genuine training mean. This simple baseline beats most elaborate models on CMU data — establish it before adding complexity.
**Protocol.** Train on sessions 1–2, test on 3–4. **Cross-session is mandatory**; same-session numbers are optimistic by 2–5×.
**Metrics.** FAR, FRR, EER, ROC-AUC, and **FRR at FAR = 0.1%**.

**Extensions that make this original rather than a replication:**
- **Quantisation sweep.** Artificially round all timestamps to 0.1 / 1 / 2 ms and re-run. Produces the "how much does timer coarsening cost?" curve, which the pre-2018 literature does not have.
- **Motor-chunk features (A2).** Add chunk boundary vector, within/between-chunk latency ratio, boundary stability. Test whether they add information over the 28-dim baseline — and whether they are *more* robust under quantisation.
- **Informed impostor.** Have impostors type the *known* password rather than using other users' data as impostor samples. This is a far stronger and more realistic attacker.

**Effort:** ~2 days of implementation plus the collection sessions. **Start here** — it produces a hard number fastest.

---

## Experiment 3 — The cross-device / cross-person 2×2 ★

**Question.** Can we distinguish two humans when the device fingerprint is *identical* — and can we avoid flagging the same human on a new device?

**This requires deliberate design and is the reason the other experiments exist.**

**Setup.** Each subject enrols on **Device A** (≥ 10 sessions). Then collect test sessions in all four cells:

| Cell | Protocol | Expected system output |
|---|---|---|
| Same person / same device | Subject continues on Device A | **Accept** |
| Same person / **different device** | Subject repeats the task on Device B | **Accept** — must *not* be flagged |
| **Different person** / same device | A labmate performs the task on the subject's own machine and browser profile | **Flag** — the whole point |
| Different person / different device | Labmate on their own machine | **Flag** — easy case |

**Recruiting note.** Pair subjects — each person is both a subject and the impostor for their partner. This halves recruitment and makes the third cell natural ("swap laptops for five minutes").

**Analysis.**
- Confusion matrix **per axis**, not just combined.
- **The headline number: behavioural EER conditioned on identical device fingerprint.** This is the paper.
- False-flag rate in the cross-device-same-person cell — the usability cost.
- Whether the **axis-disagreement pattern** predicts the true cell better than the fused scalar does.

**Effort:** ~2 weeks including recruitment. **Highest research value in the plan.**

---

## Experiment 4 — Free-text continuous authentication and detection delay

**Question.** How fast can an impostor be detected mid-session?

**Setup.** A 60–120 s task: transcribe a paragraph, then navigate a mock UI (form fill, scroll, click targets). Sliding window of 100 keystrokes / 30 s of pointer activity.

**Analysis.**
- **EER versus window size** — the curve, not a point.
- **Detection delay**: keystrokes and seconds to flag at fixed FAR = 0.1%.
- Add **A3 spectral features** (tremor band power, spectral centroid of pointer velocity) and measure the marginal gain over time-domain features.
- **Cross-task split**: train on transcription, test on UI navigation, to check whether features overfit the page.

**Bootstrap option.** Pre-train an embedding on the **Aalto 136M** dataset, fine-tune per user on local data. Tests whether public breadth substitutes for local depth — directly extending the Siamese breadth/depth paper in [[big_idea/02 Papers]].
**Effort:** ~1 week, reusing Experiment 2's collector.

---

## Experiment 5 — Active challenges and adversarial red-team

**Question.** How much signal does a controlled probe buy per second of friction, and what survives a motivated attacker?

**Part A — challenge efficacy.** Implement A2, A5 and A6 from [[big_idea/08 Active Challenge Designs]]. For each, measure *identity information per second of user friction* and compare against passive observation over the same duration. **A5 scrambled keypad** is the most interesting: it should be measurably replay-proof.

**Part B — red team.** Have subjects attempt to defeat the system:

| Attack | Tests |
|---|---|
| UA-spoofing and canvas-noise extensions | Fingerprint robustness and consistency checks |
| VPN, Tor, datacenter IP | Context layer |
| Firefox `resistFingerprinting`, Safari private mode, Brave | Signal availability under hardened browsers |
| Headless Chrome with stealth patches | Automation detection |
| **Mimicry** — impostor watches the victim type, then imitates | The realistic behavioural ceiling |
| **Replay** — captured timing replayed programmatically | Low-variance detection and A5 randomisation |

**Deliverable.** A signal-survival table: which of the ~40 attributes and ~30 behavioural features remain useful under each attack. **This table is what determines the production feature set.**
**Effort:** ~1 week.

---

> [!note] Superseded
> The week-by-week build order below assumed the fingerprint collector would be
> written first. It was not — the keystroke harness came first, and its timing was
> verified against the kernel, which this plan did not anticipate. See
> [[big_idea/13 Roadmap]] for the current sequence. **The experiment designs and
> evaluation protocols above stand unchanged.**

## Build order (superseded — see [[big_idea/13 Roadmap]])

```
Week 1   Collector page (fork FingerprintJS v3) + event logging + server JA4 capture
         → launch Experiment 1's 4-week collection window immediately; it runs in background
Week 1–2 Experiment 2: fixed-text keystroke baseline + quantisation sweep
Week 2   Add A2 motor-chunk features; compare against the 28-dim baseline
Week 3–4 Experiment 3: the 2×2 ★
Week 4   Experiment 1 analysis (collection window closes)
Week 5   Experiment 4: free-text + detection delay
Week 6   Experiment 5: active challenges + red team
Week 7   Fusion model, calibration at realistic base rates, write-up
```

> [!note] Start the collection window on day one
> Experiments 1 and 3 are gated by wall-clock time, not by implementation effort. Get the collector live before building any model — drift and cross-session data cannot be manufactured later.

## Practical stack

| Component | Choice | Rationale |
|---|---|---|
| Fingerprint collector | Fork **FingerprintJS v3 (OSS)** | Mature, ~30 attributes, extend from [[big_idea/07 Browser Signal Catalogue]] |
| Consistency reference | **CreepJS** | Best available reference for spoof/inconsistency detection |
| Server fingerprint | **JA4+ (FoxIO)** behind a TLS-terminating proxy you control | Unspoofable axis; needs raw ClientHello access |
| Passkeys | **SimpleWebAuthn** | Well-documented; implements Layer A |
| Event capture | Raw `keydown`/`keyup` with `event.code`; `pointermove` with `getCoalescedEvents()` | Never log characters |
| Storage | Raw event streams **separate from** derived features | You will re-extract features after the first modelling pass |
| Baseline model | Scaled Manhattan distance | Justify any complexity against it |
| Comparison vendor | **TypingDNA** free tier | External reference point for keystroke performance |

## Ethics checklist before collecting anything

- [ ] Informed consent naming behavioural biometric collection **explicitly** (GDPR Art. 9 / DPDP)
- [ ] `event.code` only — **no characters logged**, stated in the consent form
- [ ] Subjects use a **throwaway password**, never a real credential
- [ ] Stated retention period and a working deletion route
- [ ] Withdrawal without penalty, including retroactive data deletion
- [ ] Institutional ethics review where required
- [ ] Explicit note that the third cell of Experiment 3 involves a labmate using the subject's own machine — consent from **both** parties
- [ ] Record input modality and assistive technology per subject, so per-subgroup FRR can be reported

## Related notes

- [[big_idea/05 Datasets and Data Availability|Datasets and data availability]] — the collection schema
- [[big_idea/06 Frameworks and Validation|Frameworks and validation]] — metrics and attacker models
- [[big_idea/08 Active Challenge Designs|Active challenge designs]] — the probes used in Experiment 5
