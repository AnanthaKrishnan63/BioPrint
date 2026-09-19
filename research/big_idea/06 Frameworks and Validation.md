---
title: Account Holder Verification — Frameworks and Validation
tags:
  - big-idea
  - identity
  - research/validation
updated: 2026-09-18
---

# Frameworks and validation

## Summary

Three things decide whether this system is any good, and none of them is the choice of classifier: **how the axes are fused**, **what operating point is chosen given a realistic base rate**, and **how the template adapts as the user changes**. Systems fail in production not because the model is weak but because a 1% false-accept rate produces a hundred false alarms per true detection, or because a user who bought a new keyboard is locked out for a month.

---

## Fusion architecture

Do **not** build one monolithic classifier over all signals. Build per-axis scores, then fuse.

```
S_bind     = cryptographic binding present and valid?        (Layer A) → short-circuits
S_device   = fuzzy-match score against known device records  (Layer B)
S_context  = risk from IP / ASN / geo / login hour           (Layer C)
S_behavior = per-user verifier score                         (Layer D)
```

**Fuse at score level as log-likelihood ratios:**

```
LLR_i = log( P(s_i | genuine) / P(s_i | impostor) )
Total = Σ_i w_i · LLR_i
```

Why LLR rather than a single end-to-end model:
- **Graceful degradation.** No keyboard input on this page yet → drop that term rather than imputing a value. A monolithic model must be fed something, and imputation silently biases the score.
- **Interpretability.** You can state *which* axis fired, which is what determines the correct response.
- **Independent calibration.** Each axis has a different data volume and a different drift rate.

> [!important] The most informative feature is axis disagreement
> Collapsing everything into one number throws away the diagnosis. Report the pattern:
> - device ✓ / behaviour ✗ → **"known machine, unfamiliar hands"** → local stranger or RAT → *invisible* challenge (A1, A2)
> - device ✗ / behaviour ✓ → **"new machine, familiar hands"** → legitimate upgrade → *low-friction* confirmation
> - device ✗ / behaviour ✗ → remote attacker → *full* step-up
> The right response differs in each case. A single scalar cannot express this.

---

## Per-user modelling

You have **positive data only** — genuine sessions. There is no labelled impostor data for a specific account.

| Stage | Sessions | Approach |
|---|---|---|
| **Cold start** | 0–5 | Device + context only. No behavioural claim. |
| **Warm** | 5–30 | Per-user verifier trained with a **global impostor pool** drawn from other users' sessions. This substantially outperforms pure one-class methods and is the standard trick. Alternatives: one-class SVM, isolation forest, Mahalanobis distance to the user's mean. |
| **Mature** | 30+ | **Siamese / triplet embedding** trained across all users, then a per-user centroid and threshold. Handles new users without retraining, which is the main operational advantage. |

> [!note] Do not skip the simple baseline
> On the CMU benchmark, **scaled Manhattan distance** to the genuine mean beats most later deep methods. Establish it first; justify any added complexity against it.

---

## Decision bands and response

Never binary. Risk drives **friction**, not denial.

| Band | Action |
|---|---|
| High confidence | Proceed silently |
| Medium | Silent monitoring; require re-auth only for sensitive actions (payment, password change, data export, adding a recipient) |
| Low | Step-up: passkey, OTP, or an [[big_idea/08 Active Challenge Designs\|active challenge]] |
| Very low | Block **and** notify the owner out-of-band |

> [!note] Type I and Type II errors, named for this problem
> The null hypothesis is **"this session is the account holder."**
>
> | | Truth: account holder | Truth: someone else |
> |---|---|---|
> | **System says: holder** | ✅ correct accept | **Type II error** — false accept. Miss. Rate = **FAR**. The security cost. |
> | **System says: someone else** | **Type I error** — false reject. False alarm. Rate = **FRR**. The usability cost. | ✅ correct reject |
>
> The two are traded against each other by the threshold; **EER** is the point where they are equal. EER is a model-comparison number and is *not* a sensible operating point here, because the two errors have wildly asymmetric costs and the base rate is extreme. A Type I error costs a genuine user a few seconds of step-up friction. A Type II error costs an account. But because genuine users outnumber impostors by 10⁴–10⁶ to one, **a small Type I rate produces far more absolute incidents than a large Type II rate** — which is why the response must be friction rather than denial.

> [!warning] Base rates dominate everything
> Genuine impostor logins are roughly **1 in 10⁴–10⁶**. Worked at the *favourable* end of that range — 1,000,000 logins/day, impostor rate 1 in 10⁴, so ~100 impostors and ~999,900 genuine:
>
> | Operating point | False alarms/day (genuine users flagged) | True detections/day | Ratio |
> |---|---|---|---|
> | FRR = 1% | ~9,999 | ~99 | **~100 : 1** |
> | FRR = 0.1% | ~1,000 | ~99 | ~10 : 1 |
> | FRR = 0.01% | ~100 | ~99 | ~1 : 1 |
>
> At the unfavourable end (1 in 10⁶) every ratio worsens by 100×. Consequences:
> - Report **FRR at FAR ≤ 0.1%**, not EER, as the headline operating metric: fix the security cost at something tolerable, then state the user pain it buys. EER is a model-comparison number, not an operating point.
> - Design the response as friction rather than denial, so a false alarm costs a few seconds instead of an account.
> - Measure the **challenge rate on genuine users** — if more than a few percent of legitimate logins get challenged, the system will be switched off.

---

## Template drift and adaptation

The most-cited cause of production failure, and the subject of a dedicated 2025 survey (see [[big_idea/02 Papers]]).

**Update rule.** After a session that ends with *confirmed* genuine evidence — a successful step-up, no complaint, no chargeback — update with exponential decay:

```
μ ← (1 − α)·μ + α·x        α ≈ 0.05
```

**Guards:**
- Update **only** on confirmed-genuine sessions, never on merely-unchallenged ones, or an attacker with persistent access gradually poisons the template.
- Cap the per-update movement so one anomalous session cannot shift the template far.
- Keep an immutable enrolment anchor and periodically check that the drifted template has not wandered too far from it.
- Device records need drift handling too — accept 0.60–0.85 similarity as "device evolved" and update the stored vector, keeping the full history so gradual drift chains correctly.

---

## Evaluation protocol

> [!important] Protocol, not architecture, determines the reported number
> Published EERs below ~0.1% in this field are almost always same-session. The protocol below is what makes a result meaningful.

**Splits, in increasing order of honesty:**

| Split | What it tests | Typical inflation |
|---|---|---|
| Random within session | Nothing useful | 5–10× optimistic |
| **Cross-session** (train sessions 1–2, test 3–4, different days) | Day-to-day stability | **Minimum acceptable** |
| **Cross-device** | Keyboard/mouse/screen independence | Realistic |
| **Cross-browser** | Timer-coarsening robustness | Realistic |
| **Cross-task** | Whether features overfit the page layout | Strictest |

**Metrics to report:**

| Metric | Why |
|---|---|
| FAR, FRR, **EER** | Standard comparison |
| **FRR at FAR = 0.1%** | The actual operating point |
| ROC-AUC and the full ROC | Threshold-independent view |
| **Detection delay** (keystrokes / seconds to flag at fixed FAR) | Determines whether continuous auth is usable at all |
| **EER conditioned on identical device fingerprint** | **The number this project exists to produce** |
| Genuine-user challenge rate | Usability cost |
| Per-subgroup FRR | Equity — see below |

**Attacker models to evaluate against, in increasing strength:**

1. **Zero-effort impostor** — another user's sessions used as impostor data. The standard, and the weakest.
2. **Informed impostor** — knows the password, types it naively. Tests A2 chunk structure directly.
3. **Observant impostor** — has watched the victim type (video, shoulder-surf). The realistic ceiling.
4. **Replay attacker** — has captured raw timing and replays it. Defeated only by randomised challenges (A5) and low-variance detection.
5. **Synthetic/generative impostor** — timing generated by a model trained on the victim. The frontier threat; the synthesised-keystroke dataset in [[big_idea/05 Datasets and Data Availability]] is the right test set.

---

## Failure modes to design around

| Failure | Mitigation |
|---|---|
| Browser auto-updates change canvas/UA-CH/GPU strings monthly | Fuzzy matching with drift acceptance; never hash to a single ID |
| Shared devices, family accounts, shared workstations | Support multiple behavioural profiles per account; do not assume one account = one human |
| Mobile keyboards destroy keystroke timing | Switch to motion + touch features on mobile; maintain **separate models per platform** |
| Timer coarsening differs per browser | Per-browser-family normalisation; prefer quantisation-robust features (ranks, ratios) |
| VPN use is normal and rising | Never treat VPN as guilt; weight ASN *familiarity* over ASN *class* |
| Injury, new keyboard, fatigue, medication, time of day | Template adaptation plus a wide medium band that monitors rather than blocks |
| Attacker poisons the template through persistent access | Update only on confirmed-genuine sessions; cap per-update movement |

---

## Equity and accessibility

> [!warning] This is a correctness requirement, not a compliance footnote
> Screen readers, switch access, voice control, eye-tracking, tremor conditions, one-handed use, and non-native keyboard layouts all produce behavioural distributions far from the training median. A model trained on median users will systematically **false-reject disabled users** — the group least able to absorb the friction.
>
> - Report **FRR broken down by input modality and assistive technology**, not just in aggregate.
> - Always provide a **non-behavioural path** to the same assurance level. A passkey provides higher assurance than any behavioural signal and requires no behavioural data at all — it is both the accessible path and the strong path.
> - Perturbation challenges (A1) must be disableable and must never target a destructive control.

---

## Legal and ethical constraints

| Regime | Constraint |
|---|---|
| **GDPR Art. 9** | Behavioural data processed **to uniquely identify an individual** is special-category biometric data. Requires explicit consent or another Art. 9 condition; "legitimate interest" alone does **not** suffice. |
| **GDPR Art. 22** | Automated decisions with legal or similarly significant effects (account lockout) require human review, contestability, and explanation. |
| **India DPDP Act 2023** | Consent-and-purpose-limitation regime; notice and purpose specificity required. Relevant given the likely deployment context. |
| **Illinois BIPA** | Private right of action with statutory damages. Behavioural biometrics has been litigated under it. |
| **ePrivacy / CNIL guidance** | Fingerprinting generally requires consent. The security/fraud-prevention exemption is **narrower than commonly assumed** and does not cover open-ended profiling. |

**Architectural consequences — these are design constraints, not paperwork:**

- **Never log characters.** Store `event.code`, never `event.key`. This removes keylogging exposure entirely and is also the honest engineering choice.
- **Store templates, not raw streams,** in production. Keep raw streams only in the consented research dataset, with short retention.
- **Prefer on-device scoring.** Federated or purely client-side verification means raw behavioural data never leaves the browser — see the federated mouse-dynamics work in [[big_idea/02 Papers]]. This is the strongest available answer to Art. 9.
- **Disclose plainly, and provide opt-out** with an equivalent-assurance alternative (passkey).
- **Short retention** with documented deletion.

---

## Honest positioning

> [!important] What this system can and cannot claim
> The evidence supports **risk scoring, step-up triggering, and session-anomaly detection**. It does **not** support positively *identifying* an individual at the reliability of a possession or biometric factor.
>
> The defensible framing: **this system decides when to ask for stronger proof.** Cryptographic binding provides that proof. Behavioural and device signals decide *when to demand it* — they are a trigger, not an authenticator. Any claim beyond this overstates what the evidence will carry, and NIST SP 800-63-4 is explicit that risk indicators inform verifier decisions rather than establishing an assurance level on their own.

## Related notes

- [[big_idea/02 Papers|Papers]]
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]
- [[big_idea/08 Active Challenge Designs|Active challenge designs]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
