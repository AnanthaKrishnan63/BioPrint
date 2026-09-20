"""Build the September 20 report snapshot from saved results, without model runs.

Run from the repository root: bash scripts/research.sh reports/build_research_reports.py
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/benchmarks'
OUT = ROOT / 'reports/2026-09-20'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((SOURCE / 'summary_v11.json').read_text())
rows = summary['rows']
assert len(rows) == 72
manifest = dict(summary['source_sha256'])
extras = [
    'summary_v11.json', 'summary_v11.csv',
    'bot_rules/results.json', 'bot_rules/api_replay.json',
    'pointer_znorm_uncertainty_v1/results.json',
    'beacon_type2branch_api_replay_v1/report.json',
    'beacon_type2branch_fusion_train_v1/README.md',
    'beacon_type2branch_protocol_v1/protocol.json',
    'results/keyrecs-v1/fixed-frozen.json',
    'results/keyrecs-v1/free-frozen.json',
    'type2branch_capture_api_replay_v1/report.json',
    'brainrun_train_schema_v2/report.json',
    'cmu-account-selection/api_replay.json',
    'hmog/api_replay_v1/replay_complete.json',
    'pointer_sapimouse_znorm_dev_v2/api_replay.json',
    'pointer_sources.md',
]
for name in extras:
    manifest[name] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
for name, expected in manifest.items():
    src = SOURCE / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == expected, name
    dst = OUT / 'evidence' / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
(OUT / 'evidence/SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')

def row(dataset, model):
    return next(r for r in rows if r['dataset'] == dataset and r['model'] == model)

def rates(r):
    return ' | '.join(f"{100 * r[k]:.2f}" for k in ['far', 'frr', 'eer'])

pairs = [
    ('CMU, 10 enrollments', 'cmu-account-selection', 'baseline', 'account_selected'),
    ('CMU, 100 enrollments (separate budget)', 'cmu100', 'baseline', 'svm-1-1'),
    ('KeyRecs fixed', 'keyrecs-fixed', 'reference_logistic', 'train_selected'),
    ('KeyRecs free transcription', 'keyrecs-free', 'reference_logistic', 'train_selected'),
    ('SapiMouse, five blocks', 'pointer_sapimouse_optimized', 'scaled_manhattan@5', 'cosine@5'),
    ('FPStalker browser linkage', 'device', 'equal_attribute_agreement', 'hist_gradient_boosting'),
    ('DELBOT geometry', 'delbot', 'bioprint_pointer_rules_only', 'geometric_random_forest'),
    ('Stroop/Flanker cognitive profile', 'cognitive', 'condition_index_slope', 'full_rt_accuracy_profile'),
    ('TSI touch landing', 'touch_tsi', 'landing_baseline', 'selected'),
    ('Balabit pointer (mixed result)', 'pointer', 'existing_scaled_manhattan', 'extra_1'),
]
comparison = '| Dataset | Baseline FAR | New FAR | Baseline FRR | New FRR | Baseline EER | New EER | EER reduction (pp) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for label, dataset, old, new in pairs:
    a, b = row(dataset, old), row(dataset, new)
    vals = [a['far'], b['far'], a['frr'], b['frr'], a['eer'], b['eer'], a['eer'] - b['eer']]
    comparison += '| ' + label + ' | ' + ' | '.join(f'{v*100:.2f}' for v in vals) + ' |\n'

methodologies = {
    'cmu-account-selection': ('Scaled Manhattan; ten session-1 enrollments', 'Per-account RBF SVM; select on sessions 2–3, calibrate on session 4; same ten enrollments'),
    'cmu100': ('Scaled Manhattan; 100 enrollments', 'RBF SVM; TRAIN-selected settings and thresholds; same 100-enrollment budget'),
    'keyrecs-fixed': ('Logistic regression; 47 positional timing features', 'ExtraTrees on the same features; chronological session-1 fit/selection/calibration, session-2 DEV'),
    'keyrecs-free': ('Logistic regression; 35 summaries per 50 digraphs', 'ExtraTrees on the same summaries; session-1 selection/calibration, session-2 DEV'),
    'pointer_sapimouse_optimized': ('Handcrafted scaled Manhattan; five blocks / 641 coordinates', 'Fully convolutional sequence encoder with cosine template scoring; same observation budget'),
    'device': ('Equal agreement across browser attributes', 'Pairwise histogram gradient boosting on attribute equality/similarity; browser linkage only'),
    'delbot': ('Pointer-only production-rule reconstruction; rejects all humans at the frozen threshold', 'Random forest on pointer geometry; held-out bot-family evaluation'),
    'cognitive': ('Single condition-index reaction-time slope', 'Full reaction-time and accuracy profile; same nine DEV people, hundreds of trials per session'),
    'touch_tsi': ('Touch-landing baseline', 'Random forest with target-normalized landing, motor and timing features'),
    'pointer': ('Handcrafted scaled Manhattan', 'TRAIN-selected ExtraTrees; FRR improves but pooled EER worsens'),
    'beacon-type2branch-paired-dev': ('TypeNet + SapiMouse + handcrafted hybrid', 'Type2Branch + SapiMouse + handcrafted hybrid; same paired claims and TRAIN selection rule; FAR improves but FRR/EER worsen'),
}
improvement_pairs = pairs + [
    ('BEACON paired hybrid (FAR-only gain)', 'beacon-type2branch-paired-dev', 'old_hybrid', 'type2branch_hybrid'),
]
improvements = '| Dataset | FAR improvement (pp) | FRR improvement (pp) | EER improvement (pp) | Old statistics / methodology | New methodology |\n|---|---:|---:|---:|---|---|\n'
for label, dataset, old, new in improvement_pairs:
    a, b = row(dataset, old), row(dataset, new)
    changes = ' | '.join(f"{100 * (a[k] - b[k]):+.2f}" for k in ['far', 'frr', 'eer'])
    old_stats = ' / '.join(f"{100 * a[k]:.2f}%" for k in ['far', 'frr', 'eer'])
    old_method, new_method = methodologies[dataset]
    improvements += f'| {label} | {changes} | {old_stats}; {old_method} | {new_method} |\n'

paired = '| Model | False accepts / 162 | False rejects / 81 | FAR (%) | FRR (%) | EER (%) |\n|---|---:|---:|---:|---:|---:|\n'
for r in rows:
    if r['dataset'] == 'beacon-type2branch-paired-dev':
        paired += f"| {r['model']} | {r['false_acceptances']} | {r['false_rejections']} | {rates(r)} |\n"

appendix = '| Dataset / experiment | Model | FAR (%) | FRR (%) | EER (%) | EER aggregation | Evidence |\n|---|---|---:|---:|---:|---|---|\n'
for r in rows:
    appendix += f"| {r['dataset']} | {r['model']} | {rates(r)} | {r['eer_aggregation']} | [JSON](evidence/{r['source']}) |\n"

report = '''# BioPrint: research methods and measured results

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

IMPROVEMENTS

### Absolute rates for the dataset comparisons

COMPARISON

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

PAIRED

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

APPENDIX
'''
report = report.replace('IMPROVEMENTS', improvements).replace('COMPARISON', comparison).replace('PAIRED', paired).replace('APPENDIX', appendix)
(OUT / 'DATASET_RESULTS.md').write_text(report.rstrip() + '\n')

integration = '''# BioPrint: findings to integrate into normal login

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

PAIRED

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
'''
(OUT / 'LOGIN_INTEGRATION_PRIORITIES.md').write_text(integration.replace('PAIRED', paired))
print(f'Wrote two reports, {len(rows)} result rows and {len(manifest)} hashed evidence files to {OUT}')
