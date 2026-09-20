# BioPrint: findings to integrate into normal login

**Decision brief, 20 September 2026.** Prioritize feature-based typing and calibrated additional verification. Evaluate pointer as a separate challenger. No existing neural hybrid has earned promotion to the normal login. This document is a proposed implementation and experiment plan; no deployment or new performance measurement is implied. The [research report](DATASET_RESULTS.md) contains methodology, uncertainty and the complete 72-row ledger.

## 1. Findings ranked by practical value

| Priority | Finding and measured result | Integration decision |
|---|---|---|
| 1 | CMU account SVM: FAR **1.10 → 0.95%**, FRR **74.20 → 67.43%**, EER **22.04 → 14.32%**. FRR-selected version: **0.99 / 64.10 / 18.50%**. | Test feature-based typing first; select on the intended low-FAR operating objective. |
| 1 | KeyRecs fixed ExtraTrees: **1.26 / 76.39 / 17.08% → 1.29 / 45.91 / 11.62%** (FAR / FRR / EER). Free: **0.70 / 85.73 / 16.66% → 1.14 / 45.64 / 9.72%**. | Nonlinear timing features are promising; fixed-password and longer transcription protocols need different adapters. |
| 2 | SapiMouse cosine: **8.30 / 60.53 / 26.17% → 1.66 / 43.42 / 13.87%**, requiring **641 coordinates**, median **16.89 s** on TRAIN. | Start in shadow mode. A quick trip to the login button may not supply enough motion. |
| 2 | Browser linkage gradient boosting: **0.69 / 8.33 / 3.79% → 0.84 / 4.85 / 2.30%**. | Improve device-change routing, keeping context separate from personal identity. |
| 3 | DELBOT RF: **3.64 / 31.15 / 14.55%** on **55 bots / 459 humans**. Genuine timing-rule false flags: **15/5,100 = 0.294%**. | Preserve useful bot checks; test geometry as advisory before allowing it to reject humans. |
| Defer | Cognitive full profile: **2.78 / 77.78 / 27.78%**, versus slope **1.39 / 88.89 / 48.61%**. | More useful than slope in this dataset, but hundreds of trials do not fit a short login. |
| Defer | TSI: **1.00 / 91.85 / 34.45%**. | Too little evidence for decisive touch authentication. |

These priorities reflect transfer cost and evidence, not a numerical ranking across incompatible datasets. The **normal website currently uses scaled-Manhattan typing, device-change routing, bot rules and conditional keypad verification; pointer is advisory**. It has no measured app-wide EER. A source-dataset model cannot be inserted merely because its EER looks lower.

## 2. Compatibility requirements before model promotion

**Typing:** The current form supports arbitrary passwords and variable-length positional features. CMU uses one fixed password; KeyRecs fixed uses 47 features. Their supervised classifiers require representative negative/background data in the same feature space. Do not load their account models into unrelated website accounts, pad features to force compatibility, or assume ten samples supply the separate fit/selection/calibration sessions. Implement an explicitly versioned compatible feature adapter and fit with authorized data. A population model on password-independent timing summaries is a candidate adaptation requiring new evaluation, not an already demonstrated gain.

**Pointer:** Preserve timestamps, coordinate units, displacement scaling, input length and device class. Never turn missing motion into a zero anomaly score. Return “insufficient evidence” and keep the typing path usable. The SapiMouse 641-coordinate requirement must be measured against actual login captures.

**Device:** FPStalker includes attributes the current collector may not provide. Define a supported-feature/missingness contract and recalibrate it. A familiar browser should not compensate for failing behavioral evidence. Keep separate behavioral templates for device classes.

**Models and data:** Version models, feature schemas, thresholds and calibration provenance. Use an isolated evaluation database; do not train against or mutate `bioprint.db` during replay. Keep existing account templates readable, and require an explicit enrollment/migration route when a new model needs more support. Never adapt a template solely because an uncertain attempt was accepted.

## 3. Four combinations worth testing

Freeze these candidates before reading a new paired DEV collection. Use the same accounts, enrollment budget, attempts and capture duration for every combination.

| Candidate | Combination | Purpose | Current app FAR / FRR / EER |
|---|---|---|---|
| C0 | Current typing + bot + device routing + conditional keypad; advisory pointer | Unchanged control | Unavailable / unavailable / unavailable |
| C1 | Compatible feature-based typing challenger + unchanged routing | Isolate typing improvement | Unavailable / unavailable / unavailable |
| C2 | C1 + separate pointer score used to request additional verification | Test practical pointer value and coverage | Unavailable / unavailable / unavailable |
| C3 | TRAIN-calibrated keyboard–pointer logistic fusion, with device/bot routing separate | Test whether learned combination adds value | Unavailable / unavailable / unavailable |

Use one fixed ExtraTrees setting and the small nine-setting RBF-SVM grid as typing candidates only where the input contract and background examples are valid. Fit C3 using regularized logistic regression with **C ∈ {0.01, 0.1, 1}**, TRAIN-only scaling and an explicit missing-signal policy. Restrict candidate count; ten hours is better spent testing a few compatible models than searching many variants on the same DEV users.

Start challengers in **shadow mode**: compute and log their scores alongside the current decision without changing access. Allow additional verification for uncertain evidence; do not let an unvalidated challenger silently deny access. After validation, promote only a frozen candidate with a reversible feature flag and known model version.

## 4. What the paired hybrid experiments actually say

| Model | False accepts / 162 | False rejects / 81 | FAR (%) | FRR (%) | EER (%) |
|---|---:|---:|---:|---:|---:|
| typenet | 0 | 81 | 0.00 | 100.00 | 35.80 |
| sapimouse | 108 | 46 | 66.67 | 56.79 | 59.26 |
| handcrafted_behavior | 4 | 79 | 2.47 | 97.53 | 42.90 |
| old_dual_neural | 89 | 48 | 54.94 | 59.26 | 59.26 |
| old_hybrid | 74 | 25 | 45.68 | 30.86 | 38.27 |
| type2branch | 19 | 65 | 11.73 | 80.25 | 43.21 |
| type2branch_pointer | 105 | 48 | 64.81 | 59.26 | 59.26 |
| type2branch_hybrid | 40 | 55 | 24.69 | 67.90 | 40.74 |


These are **BEACON research results**, not C0–C3 or website metrics. Eight matched comparators use only **three DEV people / 243 claims**. The Type2Branch hybrid reduces FAR from **45.68% to 24.69%**, but FRR rises **30.86% to 67.90%** and EER rises **38.27% to 40.74%**. That is **34 fewer false accepts and 30 more false rejects**. It is not the best default merely because it is newer.

The earlier handcrafted keyboard–mouse fusion gives **FAR 2.47%, FRR 98.77%, EER 44.14%**; adding context gives **8.02%, 86.42%, 43.21%**. Nonlinear search retained logistic regression. HMOG typing+IMU fusion gives conditional **0%, 100%, 45.16%**, with only **20.13% probe coverage**, so it is not a successful full-cohort alternative. Neither corpus supplies all app features.

## 5. Other results and reasons to defer them

| Experiment | FAR / FRR / EER (%) | Meaning for this deadline |
|---|---|---|
| CMU 100-enrollment SVM | **1.03 / 44.43 / 9.14** | Improved separation costs ten times the enrollment repetitions. |
| KeyRecs TypeNet | **1.17 / 84.08 / 20.23** | Does not beat feature-based free-text typing. |
| Type2Branch short capture | **9.14 / 30.38 / 13.93** | Covers 79/79 accounts but misses the 1% FAR target; 25–100 events and five 100-event enrollment windows differ from a brief password. |
| SapiMouse Z-normalization | **0.74 / 60.53 / 11.73** | Lower FAR/EER, higher FRR; not TRAIN-selected. EER delta interval includes zero. |
| Balabit ExtraTrees | **1.93 / 92.45 / 56.18** | Pooled EER worsens from 48.61%; macro improvement is not universal success. |
| Exact browser fingerprint | **0.00 / 91.33 / 47.73** | Exact equality rejects browser changes too aggressively. |
| Cognitive condition means | **0.00 / 100.00 / 31.94** | Reject-all operating point; lower EER alone does not make it useful. |
| Cognitive learned RF | **4.17 / 88.89 / 30.56** | Worse than the simpler full profile. |
| Three-level arithmetic | **Unavailable / unavailable / unavailable** on DEV | One invalid block violates the frozen protocol. TRAIN calibration FRR is 91.67% at zero observed FAR. |
| HMOG touch extension | **Unavailable / unavailable / unavailable** | 77 support taps versus 80 required for one account. |
| BrainRun | **Unavailable / unavailable / unavailable** | Schema audit only: 101 games and 982 gestures from seven TRAIN users. |
| Original fixed-window Type2Branch | **Unavailable / unavailable / unavailable** | Required event prefix unavailable; preserved separately from the later short-capture result. |

The [full ledger](DATASET_RESULTS.md#5-evidence-and-complete-result-ledger) lists all 72 rows, including control variants and the conditional HMOG branches. Nothing omitted from this priority table should be interpreted as a successful unreported deployment.

## 6. Ten-hour execution plan

These are **proposed time boxes and acceptance criteria**, not measured run times or promised accuracy. The user's chosen priority is **lower impostor acceptance, with additional verification for uncertain users**.

| Elapsed hours | Work | Deliverable / stop condition |
|---|---|---|
| 0–1 | Freeze C0–C3, metrics, roles and feature contracts; begin authorized paired capture | Versioned protocol; reject incompatible adapters early. |
| 1–3 | Implement typing challenger and normal-login shadow path | Existing login unchanged; model/schema/missing-input tests pass. |
| 3–4.5 | Add pointer coverage and independent device routing diagnostics | Same-event timestamps verified; insufficient captures abstain. |
| 4.5–6 | Fit candidates and calibrate using TRAIN roles only | Freeze models, thresholds and selected combination before DEV. |
| 6–8 | Evaluate fresh paired DEV attempts and actual API/browser paths | Per-candidate counts, FAR/FRR/EER, coverage, latency and step-up rates. |
| 8–9 | Inspect confidence intervals and failure cases without retuning DEV | Promote only if predefined criteria pass; otherwise retain shadow mode. |
| 9–10 | Regression checks, rollback rehearsal, report and commit | Reproducible artifacts, exact model versions and a working demo. |

Begin enrollment/capture early and keep global TRAIN identities distinct from evaluation identities when feasible. Split repeated sessions into enrollment support and later probes. If the collection cannot support those roles in ten hours, report the limitation and retain the baseline. Do not manufacture a held-out result by splitting correlated windows at random.

## 7. Selection, metrics and promotion gates

Select the candidate on TRAIN to minimize `FRR(t)` subject to `FAR(t) ≤ 1%`, with fixed tie-breaking. Report both **1% and 5% TRAIN-calibrated operating points**, plus diagnostic EER, on frozen DEV. The 5% point shows the tradeoff; it does not replace the security objective.

For every candidate report `FA/N_impostor`, `FR/N_genuine`, subject/device/session counts, missing-signal coverage, conditional error rates, genuine nonacceptance including abstentions, step-up frequency/completion and end-to-end latency p50/p95. Proposed p95 added server-scoring budget: **100 ms**, to be measured separately from capture time. No latency claim is currently established for these candidate login combinations.

Require **zero score/decision mismatches** between offline and API implementations at frozen thresholds, plus successful browser typing/click/enrollment/step-up tests. Unit and API parity tests establish correctness, not recognition quality. Include mobile virtual keyboards, missing pointer data, new devices, stale models and absent calibration profiles.

Use participant/session-clustered paired uncertainty where supported. With zero false accepts, an independent-trial one-sided 95% upper bound is `1 − 0.05^(1/n)`: **72 trials only support 4.08%**, and **299 are needed for <1%**. Correlated attacks do not provide that many independent observations. If the available data cannot support a confident low-FAR claim, keep the candidate in shadow mode rather than describing empirical 0% as proof of security.

For a multi-stage app, do not call every step-up a rejection. Measure eventual unauthorized access over attempted attacks, final genuine failure over genuine attempts, and step-up rate separately. EER requires a defined scalar score and threshold sweep; the existing branching rules have no established single app EER. Record a continuous behavioral score for candidate fusion, while reporting the complete policy's operating FAR/FRR separately.

A simple decision-cost expression is `L = c_FA·π_I·FAR + c_FR·π_G·FRR + c_step·P(step-up)`. Costs and attack prevalence must be specified; the artificial genuine/impostor ratio in a dataset is not real prevalence. On the matched BEACON claims, switching hybrids changes error cost by `−34·c_FA + 30·c_FR`, before step-up costs. This explains a tradeoff; it does not establish which model is safe for deployment.

## 8. Evidence snapshot

Both reports share [precise metric data](evidence/summary_v11.json), [CSV](evidence/summary_v11.csv) and [checksums](evidence/SHA256.json). Copying reports does not enable any research model in the website. The next concrete implementation milestone is **a compatible typing challenger with paired measurement and reversible shadow scoring**, followed by evidence-based selection among C0–C3.
