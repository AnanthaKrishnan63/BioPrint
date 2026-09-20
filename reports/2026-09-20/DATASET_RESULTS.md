# BioPrint: research methods and measured results

**Evidence snapshot: 20 September 2026.** This report covers 72 DEV result rows from 25 frozen reports, including three explicitly infeasible experiments. It records improvements, tradeoffs and failures. Results describe public-dataset experiments, not the current website's authentication accuracy. The [companion integration report](LOGIN_INTEGRATION_PRIORITIES.md) turns these findings into a ten-hour work plan.

## 1. How to read the numbers

All table rates are percentages. Lower is better. FAR and FRR are measured at thresholds calibrated on TRAIN with a **1% FAR target**, unless stated otherwise. The achieved DEV FAR may exceed 1%. EER is a separate diagnostic threshold sweep; it is not the operating FAR or FRR. “pp” means percentage points. Missing results mean **unavailable**, never zero.

For a distance score d, accept when d ≤ t:

```text
FAR(t) = accepted impostor claims / all impostor claims
FRR(t) = rejected genuine claims / all genuine claims
t* = largest permitted TRAIN-calibration threshold with empirical FAR(t*) ≤ 0.01
EER ≈ FAR(t_EER) ≈ FRR(t_EER), with the source's discrete/interpolated convention
absolute improvement = old error − new error
relative improvement = (old error − new error) / old error
```

Reverse the inequality for similarity scores. CMU and KeyRecs average per-account EERs; most other experiments pool claims. These quantities cannot be averaged into an app-wide EER. The appendix preserves each aggregation convention.

Fit, model selection and threshold calibration use separate TRAIN roles where specified. DEV measures frozen candidates. First-session enrollment for an unseen account is support data, not a probe. Experiments preserve earlier failures and disclose prior DEV exposure. Legacy CMU exploration prevents a pristine, untouched-test claim. This report reads saved artifacts only; it does not run models or access test observations.

## 2. Dataset improvements and tradeoffs

### Percentage-point improvements and methodology

**Improvement = old rate − new rate**, calculated before rounding. Positive values mean fewer errors; negative values mean more errors. Old statistics are **FAR / FRR / EER**, in that order. EER means equal error rate. This table includes datasets with improvement in at least one of these metrics; a row does not imply improvement in all three. CMU's 100-enrollment experiment is a separate budget. BEACON is a paired research hybrid, not the normal website's all-feature result. Experiments without valid recognition metrics are excluded.

| Dataset | FAR improvement (pp) | FRR improvement (pp) | EER improvement (pp) | Old statistics / methodology | New methodology |
|---|---:|---:|---:|---|---|
| CMU, 10 enrollments | +0.15 | +6.76 | +7.72 | 1.10% / 74.20% / 22.04%; Scaled Manhattan; ten session-1 enrollments | Per-account RBF SVM; select on sessions 2–3, calibrate on session 4; same ten enrollments |
| CMU, 100 enrollments (separate budget) | -0.01 | +5.35 | +4.85 | 1.01% / 49.78% / 13.99%; Scaled Manhattan; 100 enrollments | RBF SVM; TRAIN-selected settings and thresholds; same 100-enrollment budget |
| KeyRecs fixed | -0.04 | +30.47 | +5.46 | 1.26% / 76.39% / 17.08%; Logistic regression; 47 positional timing features | ExtraTrees on the same features; chronological session-1 fit/selection/calibration, session-2 DEV |
| KeyRecs free transcription | -0.44 | +40.08 | +6.94 | 0.70% / 85.73% / 16.66%; Logistic regression; 35 summaries per 50 digraphs | ExtraTrees on the same summaries; session-1 selection/calibration, session-2 DEV |
| SapiMouse, five blocks | +6.64 | +17.11 | +12.30 | 8.30% / 60.53% / 26.17%; Handcrafted scaled Manhattan; five blocks / 641 coordinates | Fully convolutional sequence encoder with cosine template scoring; same observation budget |
| FPStalker browser linkage | -0.15 | +3.49 | +1.49 | 0.69% / 8.33% / 3.79%; Equal agreement across browser attributes | Pairwise histogram gradient boosting on attribute equality/similarity; browser linkage only |
| DELBOT geometry | -3.64 | +68.85 | +35.95 | 0.00% / 100.00% / 50.50%; Pointer-only production-rule reconstruction; rejects all humans at the frozen threshold | Random forest on pointer geometry; held-out bot-family evaluation |
| Stroop/Flanker cognitive profile | -1.39 | +11.11 | +20.83 | 1.39% / 88.89% / 48.61%; Single condition-index reaction-time slope | Full reaction-time and accuracy profile; same nine DEV people, hundreds of trials per session |
| TSI touch landing | +0.00 | +3.65 | +2.30 | 1.00% / 95.51% / 36.75%; Touch-landing baseline | Random forest with target-normalized landing, motor and timing features |
| Balabit pointer (mixed result) | -0.04 | +5.76 | -7.58 | 1.89% / 98.20% / 48.61%; Handcrafted scaled Manhattan | TRAIN-selected ExtraTrees; FRR improves but pooled EER worsens |
| BEACON paired hybrid (FAR-only gain) | +20.99 | -37.04 | -2.47 | 45.68% / 30.86% / 38.27%; TypeNet + SapiMouse + handcrafted hybrid | Type2Branch + SapiMouse + handcrafted hybrid; same paired claims and TRAIN selection rule; FAR improves but FRR/EER worsen |


### Absolute rates for the dataset comparisons

| Dataset | Baseline FAR | New FAR | Baseline FRR | New FRR | Baseline EER | New EER | EER reduction (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CMU, 10 enrollments | 1.10 | 0.95 | 74.20 | 67.43 | 22.04 | 14.32 | 7.72 |
| CMU, 100 enrollments (separate budget) | 1.01 | 1.03 | 49.78 | 44.43 | 13.99 | 9.14 | 4.85 |
| KeyRecs fixed | 1.26 | 1.29 | 76.39 | 45.91 | 17.08 | 11.62 | 5.46 |
| KeyRecs free transcription | 0.70 | 1.14 | 85.73 | 45.64 | 16.66 | 9.72 | 6.94 |
| SapiMouse, five blocks | 8.30 | 1.66 | 60.53 | 43.42 | 26.17 | 13.87 | 12.30 |
| FPStalker browser linkage | 0.69 | 0.84 | 8.33 | 4.85 | 3.79 | 2.30 | 1.49 |
| DELBOT geometry | 0.00 | 3.64 | 100.00 | 31.15 | 50.50 | 14.55 | 35.95 |
| Stroop/Flanker cognitive profile | 1.39 | 2.78 | 88.89 | 77.78 | 48.61 | 27.78 | 20.83 |
| TSI touch landing | 1.00 | 1.00 | 95.51 | 91.85 | 36.75 | 34.45 | 2.30 |
| Balabit pointer (mixed result) | 1.89 | 1.93 | 98.20 | 92.45 | 48.61 | 56.18 | -7.58 |


An improved EER does not imply that both operating errors improved. KeyRecs, cognitive profiling and DELBOT increase FAR while reducing FRR. Balabit worsens pooled EER. These are not universal wins.

### Typing: CMU and KeyRecs

**CMU:** Replace the median/spread scaled-Manhattan baseline with class-balanced RBF SVMs. Keep ten session-1 enrollment samples per account and the same timing features. Select from nine C/kernel-width combinations on sessions 2–3; calibrate on session 4; evaluate on sessions 5–6. Account-specific selection produces EER **22.04% → 14.32%**, a **7.72 pp / 35.02% relative reduction**. FAR becomes **0.95%** and FRR **67.43%**. Global SVM selection gives EER **15.50%**, FAR **1.00%**, FRR **72.98%**.

A separate selector directly minimizes TRAIN FRR at 1% FAR: DEV FRR improves further to **64.10%**, but EER worsens to **18.50%**, with FAR **0.99%**. With 100 enrollments, baseline EER **13.99% → 9.14%**; that is a tenfold larger enrollment budget, not a fair replacement for the ten-sample result. All reported strict-threshold FRRs remain high.

**KeyRecs 2023:** Compare logistic regression with TRAIN-selected ExtraTrees. Fixed text uses **47 positional timing features**; free transcription uses **35 summaries of 50 nonoverlapping digraphs**. Chronologically divide session 1 into **50% fit / 25% selection / 25% calibration**, then evaluate session 2. Fixed text has **79 accounts / 7,896 DEV observations**; free text has **78 probe accounts / 4,525 observations**, with one enrolled account lacking a complete probe window. Fixed-text FRR falls **30.47 pp**, and free-text FRR falls **40.08 pp**. Their relative EER reductions are **31.98%** and **41.66%**. Fixed selected-candidate fit time was **1.52 s** and free **0.69 s**, excluding search and preparation.

The TypeNet architecture transfer on free transcription reaches **FAR 1.17%, FRR 84.08%, EER 20.23%** and does not beat ExtraTrees. A separate Type2Branch short-capture adaptation uses frozen length-specific thresholds and covers **79/79 accounts**: **563/6,162 impostor accepts**, **24/79 genuine rejections**, yielding **FAR 9.14%, FRR 30.38%, pooled EER 13.93%**. One probe per identity and a different capture budget prevent a direct ranking against the multi-window ExtraTrees protocol. The 1% target did not transfer across sessions.

### Pointer: SapiMouse and Balabit

**SapiMouse:** Use a trained fully convolutional sequence encoder with cosine template scoring instead of handcrafted scaled-Manhattan scoring, at the same five-block observation budget. Each decision needs **641 coordinates / 640 displacements**; training median observation time was **16.89 seconds**. The DEV cohort has **24 users**, with **76 genuine and 1,748 impostor comparisons**. False accepts fall **145 → 29**, false rejects **46 → 33**, and pooled EER falls **26.17% → 13.87%**: **12.30 pp / 46.99% relative reduction**.

An enrollment-normalized OCSVM achieves pooled EER **18.36%**, FAR **2.40%**, FRR **61.84%**. Its macro EER **7.73%** is lower than cosine's **8.27%**: the cosine result improves the pooled operating objective, not every aggregation.

A later TRAIN-background Z-normalization comparison gives **FAR 0.74%, FRR 60.53%, EER 11.73%** versus cosine's **1.66%, 43.42%, 13.87%**. This means **13 rather than 29 false accepts**, but **46 rather than 33 false rejects**. It was not selected under the TRAIN FRR objective. Its paired-bootstrap EER-difference interval includes zero; do not promote it just because its DEV EER is lower.

**Balabit:** ExtraTrees reduces FRR **98.20% → 92.45%**, but FAR rises **1.89% → 1.93%** and pooled EER worsens **48.61% → 56.18%**. Macro EER improves **38.30% → 30.09%**. Report both; the pooled result does not support deployment.

### Browser context, bots, cognition and touch

**FPStalker:** Train pairwise browser-linkage classifiers using equality and similarity attributes. Gradient boosting improves equal-agreement EER **3.79% → 2.30%** and FRR **8.33% → 4.85%**, while FAR increases **0.69% → 0.84%**. DEV has **154 browser identities**, **1,176 genuine** and **23,520 impostor pairs**. Exact fingerprint matching gives FAR **0%** but FRR **91.33%**. Learned/fuzzy matching accommodates browser changes; none of these figures proves person identity or same-device attacker detection.

**DELBOT:** Random-forest pointer geometry is evaluated on a held-out bot family: **55 bot trajectories / 459 human trajectories**. It accepts **2 bots** and rejects **143 humans**: FAR **3.64%**, FRR **31.15%**, EER **14.55%**. Pointer-only production-rule reconstruction rejects all humans at its frozen threshold: FAR **0%**, FRR **100%**, EER **50.50%**. This compares available geometry, not the entire production bot detector. Separately, production timing rules flag **15/5,100 genuine CMU samples = 0.294%**; bot-positive FAR and EER are unavailable for that genuine-only experiment.

**Stroop/Flanker:** Compare one condition-index slope with condition means and a fuller reaction-time/accuracy profile. Across **nine DEV people**, the full profile changes EER **48.61% → 27.78%**, false accepts **1/72 → 2/72**, and false rejects **8/9 → 7/9**. It uses hundreds of trials, and the three condition indices are not equally spaced difficulty levels. It does not validate a three-question arithmetic login test. A learned full-profile RF performs worse: **EER 30.56%, FAR 4.17%, FRR 88.89%**.

**TSI:** Target-normalized landing/motor/timing features with a random forest modestly improve EER **36.75% → 34.45%** and FRR **95.51% → 91.85%**, at FAR **1.00%**, over **12 DEV people / 356 trials**. These are touch landing measurements, not keypad search reaction times or hold/release measurements.

**Three-level arithmetic:** Eight TRAIN candidates compare slopes, means and fuller waiting-time/missing/error profiles. TRAIN selects the full profile with scaled L1 distance. Calibration yields **0/36 false accepts** and **11/12 false rejects (91.67%)**. DEV is infeasible: **one of 60 required blocks has 50 entries instead of five**. No DEV FAR, FRR or EER exists; no invalid block was silently truncated or dropped.

**BrainRun:** The bounded TRAIN schema audit covers **101 games / 982 gestures**, with records for **7/8 chosen TRAIN users**, including **six empty gesture arrays**. Recognition training/evaluation is not established by that audit. FAR, FRR and EER are unavailable. Blocked or unacquired sources likewise have no measured rates.

## 3. All-feature app experiments and paired combinations

**There is no measured all-feature app FAR, FRR or EER.** No acquired dataset jointly supplies the same participants' login typing, pointer, browser context, scrambled-keypad cognition and labeled bot attacks. Combining unrelated datasets by invented identity links would not create valid evidence.

The normal login uses scaled-Manhattan typing, device routing and bot rules; pointer is advisory and keypad verification is conditional. Research hybrids are separate. The following are the closest executed paired experiments, with their coverage limits.

### BEACON: matched keyboard–pointer hybrids

Use the same **140 paired windows** from **three DEV people**, yielding **81 genuine / 162 impostor claims**. Retain all **81 windows shorter than 25 keystrokes**. Freeze the encoders; fit **24 balanced logistic-regression candidates** across eight feature combinations and C in **{0.01, 0.1, 1}**. Fit on three TRAIN identities, select on two and calibrate on two others. Selection prioritizes FRR at ≤1% FAR, then EER and C. Calibration contains **72 genuine / 72 impostor claims**.

The old hybrid combines TypeNet, SapiMouse and 32 handcrafted features. The new hybrid substitutes Type2Branch for TypeNet. “All-feature” here means the available keyboard/pointer feature set; it excludes keypad cognition and labeled bot attacks and is not the whole app.

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


The Type2Branch hybrid prevents **34 additional false accepts** but causes **30 additional false rejections**: FAR improves **20.99 pp**, FRR worsens **37.04 pp**, and EER worsens **2.47 pp**. Its unweighted balanced error at the frozen threshold is **(24.69 + 67.90)/2 = 46.30%**, versus **38.27%** for the old hybrid. Balanced operating error is not generally EER.

Earlier BEACON handcrafted fusion gives **FAR 2.47%, FRR 98.77%, EER 44.14%**. Adding context gives **8.02%, 86.42%, 43.21%**. A later nonlinear search retains the same logistic model; its reported **43.83%** behavioral EER versus **44.14%** reflects interpolation convention, not improved predictions. The earlier neural-v2 hybrid has **4.32%, 92.59%, 43.21%** under its original selection; it is distinct from the newly selected matched old-hybrid comparator above.

### HMOG: paired typing and inertial sensing

Strict extraction produced **0/332 eligible windows**. A separately frozen gap-filtered protocol produced **207 windows**, with **87 fit windows / 800 training updates**. DEV supplies only **31/154 candidate genuine probes = 20.13% coverage**, and two of four DEV people lack probes. Full-cohort validation is infeasible.

| Conditional observed-probe model | FAR (%) | FRR (%) | EER (%) |
|---|---:|---:|---:|
| Joint typing + IMU | 0.00 | 100.00 | 45.16 |
| Typing branch | 0.00 | 93.55 | 26.88 |
| IMU branch | 4.30 | 96.77 | 50.54 |

The joint model accepts **0/31 genuine probes**. Its genuine nonacceptance rate including abstentions is **100%**. The touch follow-up is also infeasible: one account has **77 support taps versus 80 required**. There is no valid four-account touch-fusion EER.

### API correctness versus authentication quality

Recorded API checks reproduced **780,300 CMU scores**, **3,648 SapiMouse cosine/Z-normalized scores**, **6,241 Type2Branch short-capture scores**, **1,944 paired BEACON scores**, and **372 conditional HMOG scores**, with zero reported decision changes in their respective validations. Bot replay reproduced all **5,100 genuine flags**. These counts are separate test scopes, not independent biometric trials to pool.

BEACON replay starts from offline encoder features and exercises the real frozen fusion API. It does not test live browser-to-encoder capture. API parity proves implementation agreement, not low FAR. Arithmetic API validation reproduces an infeasible status, not biometric predictions.

## 4. What the mathematics supports

**Uncertainty:** CMU account-selected minus baseline EER is **−7.72 pp**, with paired-account bootstrap 95% interval **[−11.11, −4.55] pp**. Its FRR change is **−6.76 pp**, interval **[−14.98, +1.02] pp**. Against the global SVM, EER change **−1.18 pp** has interval **[−2.56, +0.26] pp**, while FRR change **−5.55 pp** has interval **[−10.49, −1.43] pp**. Reference models matter when quoting confidence intervals. Shared impostor probes and prior exploration limit independence.

SapiMouse Z-normalization's **−2.15 pp** observed EER difference has a 2,000-replicate participant-bootstrap interval **[−5.47, +3.98] pp**, conditional on fixed models/enrollment. This does not establish a reliable improvement.

**Low-FAR resolution:** With n independent impostor trials and zero accepts, the one-sided exact 95% upper bound is `1 − 0.05^(1/n)`. At **n = 72**, this is **4.08%**, not 1%. At least **299 zero-error independent trials** are needed for that bound to fall below 1%. Repeated claims from three people are correlated and cannot substitute for 299 independent trials. BEACON FAR moves in increments of **1/162 = 0.617 pp**; its FRR moves **1/81 = 1.235 pp**.

**Fusion:** If two modalities must both accept, joint FAR equals `P(A_keyboard ∩ A_pointer | impostor)`. Multiplying separate FARs assumes conditional independence and aligned populations; neither is established. In general, `max(0, FAR_k + FAR_p − 1) ≤ FAR_joint ≤ min(FAR_k, FAR_p)` for this AND rule on the same population. Learned fusion requires directly measured paired scores and calibration. Marginal dataset results cannot determine app EER.

**Cognitive slope:** With three equally spaced levels x = 1, 2, 3, the least-squares slope is `(mean_time_3 − mean_time_1)/2`. It discards the middle-level timing, absolute speed, variability and accuracy. A full profile retains more information, but small participant counts and long tasks limit login transfer.

## 5. Evidence and complete result ledger

The following ledger includes every row, including repeated controls and failed full-cohort HMOG diagnostics. Repeated rows are not additional independent experiments. Rates are rounded here; [the JSON snapshot](evidence/summary_v11.json) retains precision, limitations and infeasible statuses. [SHA-256 checksums](evidence/SHA256.json) bind the copied source reports. These are metric/report artifacts, not raw recordings or production account data.

| Dataset / experiment | Model | FAR (%) | FRR (%) | EER (%) | EER aggregation | Evidence |
|---|---|---:|---:|---:|---|---|
| cmu | baseline | 1.10 | 74.20 | 22.04 | macro_per_account | [JSON](evidence/cmu/dev_results.json) |
| cmu | svm-10-0.1 | 1.00 | 72.98 | 15.50 | macro_per_account | [JSON](evidence/cmu/dev_results.json) |
| cmu100 | baseline | 1.01 | 49.78 | 13.99 | macro_per_account | [JSON](evidence/cmu100/dev_results.json) |
| cmu100 | svm-1-1 | 1.03 | 44.43 | 9.14 | macro_per_account | [JSON](evidence/cmu100/dev_results.json) |
| cmu-account-selection | baseline | 1.10 | 74.20 | 22.04 | macro_per_account | [JSON](evidence/cmu-account-selection/dev_results.json) |
| cmu-account-selection | svm-10-0.1 | 1.00 | 72.98 | 15.50 | macro_per_account | [JSON](evidence/cmu-account-selection/dev_results.json) |
| cmu-account-selection | account_selected | 0.95 | 67.43 | 14.32 | macro_per_account | [JSON](evidence/cmu-account-selection/dev_results.json) |
| cmu-far-selection | baseline | 1.10 | 74.20 | 22.04 | macro_per_account | [JSON](evidence/cmu-far-selection/dev_results.json) |
| cmu-far-selection | svm-10-0.1 | 1.00 | 72.98 | 15.50 | macro_per_account | [JSON](evidence/cmu-far-selection/dev_results.json) |
| cmu-far-selection | account_selected | 0.95 | 67.43 | 14.32 | macro_per_account | [JSON](evidence/cmu-far-selection/dev_results.json) |
| cmu-far-selection | far_selected | 0.99 | 64.10 | 18.50 | macro_per_account | [JSON](evidence/cmu-far-selection/dev_results.json) |
| keyrecs-fixed | reference_logistic | 1.26 | 76.39 | 17.08 | macro_per_account | [JSON](evidence/results/keyrecs-v1/fixed-results.json) |
| keyrecs-fixed | train_selected | 1.29 | 45.91 | 11.62 | macro_per_account | [JSON](evidence/results/keyrecs-v1/fixed-results.json) |
| keyrecs-free | reference_logistic | 0.70 | 85.73 | 16.66 | macro_per_account | [JSON](evidence/results/keyrecs-v1/free-results.json) |
| keyrecs-free | train_selected | 1.14 | 45.64 | 9.72 | macro_per_account | [JSON](evidence/results/keyrecs-v1/free-results.json) |
| keyrecs-free-typenet | frozen_typenet | 1.17 | 84.08 | 20.23 | macro_per_account | [JSON](evidence/results/typenet-keyrecs-v1/dev-results.json) |
| device | equal_attribute_agreement | 0.69 | 8.33 | 3.79 | pooled | [JSON](evidence/device_results.json) |
| device | exact_fingerprint_match | 0.00 | 91.33 | 47.73 | pooled | [JSON](evidence/device_results.json) |
| device | bioprint_overlap_device_score | 1.21 | 33.08 | 7.57 | pooled | [JSON](evidence/device_results.json) |
| device | thresholdfp_2025_pairwise_adaptation | 1.12 | 7.99 | 3.97 | pooled | [JSON](evidence/device_results.json) |
| device | fpstalker_inspired_random_forest | 0.80 | 5.27 | 2.38 | pooled | [JSON](evidence/device_results.json) |
| device | hist_gradient_boosting | 0.84 | 4.85 | 2.30 | pooled | [JSON](evidence/device_results.json) |
| delbot | geometric_random_forest | 3.64 | 31.15 | 14.55 | pooled | [JSON](evidence/delbot_results.json) |
| delbot | geometric_logistic_regression | 0.00 | 100.00 | 14.60 | pooled | [JSON](evidence/delbot_results.json) |
| delbot | bioprint_pointer_rules_only | 0.00 | 100.00 | 50.50 | pooled | [JSON](evidence/delbot_results.json) |
| cognitive | condition_index_slope | 1.39 | 88.89 | 48.61 | pooled | [JSON](evidence/cognitive_results.json) |
| cognitive | condition_means | 0.00 | 100.00 | 31.94 | pooled | [JSON](evidence/cognitive_results.json) |
| cognitive | full_rt_accuracy_profile | 2.78 | 77.78 | 27.78 | pooled | [JSON](evidence/cognitive_results.json) |
| cognitive | learned_full_profile_rf | 4.17 | 88.89 | 30.56 | pooled | [JSON](evidence/cognitive_results.json) |
| touch_tsi | selected | 1.00 | 91.85 | 34.45 | pooled | [JSON](evidence/touch_tsi_results.json) |
| touch_tsi | landing_baseline | 1.00 | 95.51 | 36.75 | pooled | [JSON](evidence/touch_tsi_results.json) |
| beacon_nonlinear | behavior_fusion_selected | 2.47 | 98.77 | 43.83 | pooled | [JSON](evidence/beacon_nonlinear_results.json) |
| beacon_nonlinear | behavior_fusion_reference_lr | 2.47 | 98.77 | 43.83 | pooled | [JSON](evidence/beacon_nonlinear_results.json) |
| beacon_nonlinear | behavior_plus_context_selected | 8.02 | 86.42 | 43.21 | pooled | [JSON](evidence/beacon_nonlinear_results.json) |
| beacon_nonlinear | behavior_plus_context_reference_lr | 8.02 | 86.42 | 43.21 | pooled | [JSON](evidence/beacon_nonlinear_results.json) |
| pointer | existing_scaled_manhattan | 1.89 | 98.20 | 48.61 | pooled | [JSON](evidence/pointer/results.json) |
| pointer | extra_1 | 1.93 | 92.45 | 56.18 | pooled | [JSON](evidence/pointer/results.json) |
| pointer_sapimouse | scaled_manhattan@5 | 8.30 | 60.53 | 26.17 | pooled | [JSON](evidence/pointer_sapimouse/results.json) |
| pointer_sapimouse | ocsvm:0.5:raw@5 | 2.23 | 80.26 | 21.28 | pooled | [JSON](evidence/pointer_sapimouse/results.json) |
| pointer_sapimouse | ocsvm:0.5:enroll@5 | 2.40 | 61.84 | 18.36 | pooled | [JSON](evidence/pointer_sapimouse/results.json) |
| pointer_sapimouse_optimized | scaled_manhattan@5 | 8.30 | 60.53 | 26.17 | pooled | [JSON](evidence/pointer_sapimouse_optimized/results.json) |
| pointer_sapimouse_optimized | ocsvm:0.5:raw@5 | 2.23 | 80.26 | 21.28 | pooled | [JSON](evidence/pointer_sapimouse_optimized/results.json) |
| pointer_sapimouse_optimized | cosine@5 | 1.66 | 43.42 | 13.87 | pooled | [JSON](evidence/pointer_sapimouse_optimized/results.json) |
| beacon | keyboard | 0.62 | 100.00 | 39.51 | pooled | [JSON](evidence/beacon/dev_results.json) |
| beacon | mouse | 41.98 | 35.80 | 40.43 | pooled | [JSON](evidence/beacon/dev_results.json) |
| beacon | behavior_fusion | 2.47 | 98.77 | 44.14 | pooled | [JSON](evidence/beacon/dev_results.json) |
| beacon | behavior_plus_context | 8.02 | 86.42 | 43.21 | pooled | [JSON](evidence/beacon/dev_results.json) |
| beacon | untrained_equal_behavior_fusion | 11.73 | 88.89 | 39.51 | pooled | [JSON](evidence/beacon/dev_results.json) |
| beacon-neural-v1 | typenet | 0.00 | 100.00 | 37.04 | pooled | [JSON](evidence/beacon-neural-v1/dev_results.json) |
| beacon-neural-v1 | mouse | 41.98 | 35.80 | 40.43 | pooled | [JSON](evidence/beacon-neural-v1/dev_results.json) |
| beacon-neural-v1 | neural_behavior_fusion | 41.98 | 34.57 | 39.51 | pooled | [JSON](evidence/beacon-neural-v1/dev_results.json) |
| beacon-neural-v1 | hybrid_behavior_fusion | 2.47 | 98.77 | 43.21 | pooled | [JSON](evidence/beacon-neural-v1/dev_results.json) |
| beacon-neural-v2 | typenet | 0.00 | 100.00 | 35.80 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| beacon-neural-v2 | mouse | 41.98 | 34.57 | 39.51 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| beacon-neural-v2 | handcrafted_behavior | 2.47 | 97.53 | 42.90 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| beacon-neural-v2 | sapimouse | 66.67 | 56.79 | 59.26 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| beacon-neural-v2 | dual_neural | 54.94 | 59.26 | 59.26 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| beacon-neural-v2 | hybrid_behavior_fusion | 4.32 | 92.59 | 43.21 | pooled | [JSON](evidence/beacon-neural-v2/dev_results.json) |
| hmog-paired-v1 | joint | 0.00 | 100.00 | 45.16 | pooled_observed_claims_incomplete_cohort | [JSON](evidence/hmog/validation_v1/validation_complete.json) |
| hmog-paired-v1 | key | 0.00 | 93.55 | 26.88 | pooled_observed_claims_incomplete_cohort | [JSON](evidence/hmog/validation_v1/validation_complete.json) |
| hmog-paired-v1 | imu | 4.30 | 96.77 | 50.54 | pooled_observed_claims_incomplete_cohort | [JSON](evidence/hmog/validation_v1/validation_complete.json) |
| pointer-sapimouse-znorm-comparison | cosine@5 | 1.66 | 43.42 | 13.87 | pooled | [JSON](evidence/pointer_sapimouse_znorm_dev_v2/dev_complete.json) |
| pointer-sapimouse-znorm-comparison | znorm_cosine@5 | 0.74 | 60.53 | 11.73 | pooled | [JSON](evidence/pointer_sapimouse_znorm_dev_v2/dev_complete.json) |
| type2branch-one-capture-dev | train_selected_length_model | 9.14 | 30.38 | 13.93 | pooled_raw_score_discrete_conditional | [JSON](evidence/type2branch_capture_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | typenet | 0.00 | 100.00 | 35.80 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | sapimouse | 66.67 | 56.79 | 59.26 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | handcrafted_behavior | 2.47 | 97.53 | 42.90 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | old_dual_neural | 54.94 | 59.26 | 59.26 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | old_hybrid | 45.68 | 30.86 | 38.27 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | type2branch | 11.73 | 80.25 | 43.21 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | type2branch_pointer | 64.81 | 59.26 | 59.26 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
| beacon-type2branch-paired-dev | type2branch_hybrid | 24.69 | 67.90 | 40.74 | pooled_discrete | [JSON](evidence/beacon_type2branch_dev_evaluation_v1/report.json) |
