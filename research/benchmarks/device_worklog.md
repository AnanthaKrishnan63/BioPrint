---
title: Device and Paired Dataset Worklog
tags: [bioprint, worklog, experiments]
updated: 2026-09-20
---

## Worklog

- Read repository instructions, device/keypad/bot implementations and hackathon judging criteria. No live database or reserved test features read.
- Searched primary dataset/paper sources. Located FPStalker's public longitudinal subset, newer cross-sectional MIT dataset, agreement-gated paired sources, and the 2026 BEACON release.
- Downloaded two FPStalker archives (143,126,719 bytes), upstream license/README/schema. Checked tar metadata: 257,121,411 uncompressed bytes; no extraction or SQL execution. Added checksummed downloader.
- Added metadata-first `device_benchmark.py prepare`: 1,096 train, 349 dev, 374 test browser IDs; 10,997 train rows, 2,063 dev rows, 1,940 opaque reserved test rows. Cached train/dev features only.
- Implemented independent FPStalker-inspired random forest, histogram boosting, exact/equal-match baselines and mapped BioPrint overlap score; train-only inner calibration and frozen dev reporting. Two CPU threads; no GPU requirement.
- Initial run revealed adapter lacked `probe_version`, making live device output unavailable. Kept invalid artifact named `device_results_v1_invalid_adapter.json`; added availability assertion, fixed adapter. The other initial model results are exploratory only; final v2 report supersedes v1.
- Located ThresholdFP (2025). Added Algorithm 2 stability weights trained on fitting identities, adapted to pair verification; explicitly not full lineage/remediation reproduction.
- BEACON manifest frozen before behavioral data download: train P002/P003/P006/P007/P008/P009/P010, dev P012/P015/P022, test P026/P028. Test files are never requested. Only first two paired sessions selected.
- Training-only feasibility check: P002/S001 keyboard begins186s; 2MiB mouse prefix ends161.9s. Revised uniformly to16MiB before dev inspection; saved v1 manifest. Planned326,990,897 bytes, pinned revision da1306428ef626108a914abff42ead19b3a46f62.
- Found stale upstream metadata on corrected BEACON files; bounded downloader records actual SHA256 and whether published full-file hash matches. Full non-mouse files capped8MiB, mouse capped16MiB, test excluded.
- Root agent owns paired BEACON feature/fusion benchmark and shared logs; this agent owns source research, download/split scripts and fingerprint benchmark.
- Fingerprint v2 completed:154 dev browser identities,1,176 genuine/23,520 impostor comparisons. EER: equal attribute agreement3.788%, mapped BioPrint overlap7.568%, ThresholdFP pair adaptation3.967%, RF2.381%, histogram boosting2.296%. At training-calibration FAR1%, boosting devFAR0.842%/FRR4.847%; this is not a deployed-person-authentication result.
- Added five passing integrity tests covering sealed-test refusal, ties/zero-FAR calibration, adapter availability and opaque invalid-UTF8 test rows never decoded.
- Corrected BEACON acquisition: a6.39MB full mouse file had retained a prior2MiB prefix. Replaced before root final training; all complete files now verify against pinned Hugging Face Git/LFS object metadata. Final60-file total326,556,916 bytes. Three published hashes stale, each verified against authoritative pinned file hash.
- Verified elapsed timestamp shared origin in published BEACON logger (Zenodo20062628): global start_time line83; keyboard elapsed line370/391; mouse line446. This supports elapsed alignment where legacy records contradict README's all-epoch claim.
- Downloaded DELBOT-Mouse4,623,382-byte MIT-repository archive;17,498,093 bytes logical uncompressed,3,551 trajectories. Split metadata by source/family before content access:1,031train,514dev,2,006sealed test. Reserved GAN/phone/fast traces never opened.
- DELBOT RF geometric baseline: unseen pyhm bot family55traces versus humanPC2 459traces givesEER14.55%; training1%FAR operating point gives devFAR3.64%/humanFRR31.15%. Logistic modelEER14.60% but calibration threshold rejects all devhumans. Existing pointer-rule subgroup hasEER50.50%. Failures retained, not promoted to deployment.
- Located directly accessible repeated cognitive task data from Hedge/Powell/Sumner2018 atOSF cwzds. Froze28train/9dev/10test subject split before download; onlyStroop/Flanker train/dev files requested (2,035,204bytes). No arithmetic/keypad equivalence claimed.
- Added explicit training-enrollment-support versus validation-probe record-role manifests for unseen-account reference observations; global weights use training cohorts only.
- Cognitive download completed: 148 SHA256-verified files, 2,035,204 bytes; 40 test-participant files never downloaded. OSF stale redirect failures resolved with explicit file-version URLs. No data exclusions in final analysis.
- Cognitive first experiment: 19 fit / 9 training-calibration / 9 dev participants. Slope EER 48.61%, condition-means 31.94%, full RT/accuracy 27.78%, learned RF 30.56%. Full profile at training-calibration FAR 1%: dev FAR 2/72 (2.78%), FRR 7/9 (77.78%). Failures remain reported; no deployment claim.
- Serialized frozen fingerprint, DELBOT and cognitive learned models, plus feature schemas and permitted dev matrices for root inference-API replay. Fingerprint/DELBOT re-fit uses identical training-only configurations solely to persist models; no dev-driven tuning.
- Added probe-browser cluster bootstrap script with 2,000 replicates and explicit conditional/dependence limitations. Added cognitive/bot representation and sealed-test guards; all 13 currently discovered benchmark-invariant tests passed before model-artifact replay.

## Judging relevance

The repository's brief prioritizes reliability (25 points). Train-calibrated FAR/FRR, leakage-controlled dev evaluation, paired fusion and reproducible scripts substantiate that criterion. Novelty or claimed perfect scores cannot substitute for genuine evaluation. The exact-task keypad and complete browser/behavior fusion remain explicit evidence gaps.

## Cognitive source and uncertainty

The source is Hedge, Powell and Sumner (2018), [The reliability paradox](https://pmc.ncbi.nlm.nih.gov/articles/PMC5990556/), with original repeated Stroop/Flanker recordings on [OSF](https://osf.io/cwzds/). The README welcomes reanalysis, but retrieved OSF metadata does not specify a license; do not infer unrestricted redistribution. Download receipts verify all 148 permitted files. Forty reserved participant files remain unfetched.

Only nine unseen participants contribute nine genuine and 72 impostor comparisons. The impostor pairs share both probe participants and enrolled references, so 72 is not an independent sample count. FAR resolution is 1/72 = 1.39 percentage points; neither a 1% nor a 0.1% operational target is established. Both requested training calibration targets select the same threshold because the calibration sample is similarly small. Zero observed accepts is not proof of zero risk.

`scripts/cognitive_uncertainty.py` performs a frozen-model sensitivity check by removing each participant from both reference and probe roles. Results are in `cognitive_uncertainty.json`. At the training-calibration 1% FAR threshold, full-profile FAR ranges from 0% to 3.57% and FRR from 75% to 87.5%; the slope's FRR ranges from 87.5% to 100%. These are deletion sensitivity ranges, not confidence intervals, and do not include model or population uncertainty. No parameters changed after this analysis.

Saved-model replay validation passed for all five learned fingerprint, bot and cognitive models. Parent API replay separately reproduced device results for 24,696 pairs across two models, with maximum probability error 3.4e-16 and identical FAR/FRR. The evidence supports reproducible research inference, not production readiness.

## Independent inference review

Reviewed the isolated modality router and both replay scripts. Found finite, oversized features could pass validation and overflow sklearn's float32 tree conversion; added explicit float32-representability rejection with HTTP 422, preserving representable values without clamping. Found BEACON replay compared loaded-model scores but did not assert saved-report metrics or returned decisions. It now checks saved dev EER, all FAR/FRR operating points, exact threshold metadata and every inclusive-threshold acceptance flag. Tabular replay also checks threshold metadata and every decision. New output paths preserve original replay reports.

Four synthetic tests in `scripts/test_modalities_review.py` pass. Stricter BEACON replay passes all 243 pairs across five models with zero score difference; cognitive replay passes all 81 pairs with maximum difference 2.22e-16. New artifacts are `beacon/api_replay_review.json` and `cognitive/api_replay_review.json`. Source inspection found normal CMU tests use synthetic fixtures or assert real-data loader refusal; research reserved-split tests exercise early guards. No accidental sealed-feature read was found in the inspected tests. This is a bounded source audit, not a claim about arbitrary commands outside the test suite.

## Additional touch-motor evidence

Parent located Google's UIST 2024 Touch Sensing Images release. Saved protocol before measurement acquisition, then verified the 28.34 MB CSV/layout against publisher Git object hashes. Implemented role-aware prefix filtering: task4 and user13–16 measurements remain sealed; user01–12 task1 enrolls/fits, task2 selects/calibrates, task3 validates. Metadata-only record roles enumerate every source trial. Source includes aligned target geometry and first-frame taps, but no release time or cognitive stimulus onset.

Three synthetic integrity/geometry tests pass. Training selected the fixed ten-feature random forest over logistic regression; dev was inspected only after frozen configuration and thresholds were saved. Dev has 356 genuine and 3,916 impostor comparisons: RF EER 34.45%, FAR 0.996% and FRR 91.85% at training-calibrated 1% FAR. The six-feature landing baseline gives EER 36.75% and FRR 95.51% at the same observed FAR. This combines an added timing representation and classifier change, not a controlled algorithm-only improvement. No matched-feature extra baseline was invented after dev results.

Strict API replay passed all 4,272 pairs, with maximum score error 3.33e-16, exact returned decisions/threshold and saved FAR/FRR parity. `touch_sources.md` documents acquisition, source license, sparse-trial exclusions, short within-visit task blocks, shared-participant dependence and remaining exact-keypad gaps. New touch data directory including metadata is 28,457,546 bytes, bringing the earlier nine-directory storage audit to approximately 672,196,121 bytes before unrelated later work. No network listener or production database was used.

## Runtime thread optimization

Investigated parent latency findings using seeded generated finite vectors only. `runtime_thread_benchmark.py` alternates one/two-job inference order over 20 repeats after three warmups; all runtime choices precede dev parity checks. Median one-query RF latency dropped from 69.33 to 17.86 ms (fingerprint), 82.12 to 20.77 ms (DELBOT), 81.10 to 21.13 ms (cognitive), and 84.98 to 21.13 ms (touch) when setting `n_jobs=1`. Generated 256-query batches also improved. These are warm local model timings under concurrent research load, not end-to-end login latency or universal hardware guarantees.

Before enabling the change, compared both settings across all 24,696 fingerprint, 514 bot, 81 cognitive and 4,272 touch dev pairs. Maximum probability difference was 3.33e-16; every acceptance decision matched at both frozen 1% and 0.1% calibration targets, with identical saved FAR/FRR. No weights, thresholds or serialized artifacts changed. The router now sets `n_jobs=1` only in memory for the four validated forest names. Runtime selection cannot improve the underlying discrimination metrics.

Five synthetic API review tests pass, including a check that loader parameters change only `n_jobs`. Strict API replays with the updated loader passed all six tabular models (forests plus unchanged HGB/logistic), preserving originals as `api_replay_singlethread.json`. `runtime_thread_api_validation.json` records unchanged before/after artifact SHA256 checks. Runtime report: `runtime_threads.json`. Representative generated KeyRecs/Balabit timings were sent to their owning agents for separate exhaustive decision-parity checks; no changes to those wrappers were made here.

## Training-motivated nonlinear fusion search

Parent requested compact nonlinear families because existing BEACON training selection EER was weak. Preregistered ExtraTrees, random forest and histogram boosting on unchanged 32D behavior and 33D behavior/context pairs with original fit/selection/calibration people. Both tracks retained the original logistic regression: training selection EER 41.18%, versus 47.06–52.94% among new candidates. No nonlinear candidate was evaluated on dev or promoted. Successful training took 33.2 seconds; an initial column-selector packaging failure is preserved.

Before dev access, detected that the generic tabular next-impostor threshold convention differed from original BEACON's observed-score convention. Kept initial `frozen.json`, corrected only calibration thresholds in `frozen_v2.json`, with identical models and training cohorts. Reused existing frozen dev pairs and asserted every retained-model acceptance decision and 1%-target FAR/FRR matches the original reference. Strict API replay passes 243 pairs across four identical selected/reference wrappers with zero score error; two generated integrity/context-isolation tests pass. No raw dev reread, test access or live-database access occurred.

This is a retained failed search, not new independent evidence or an improvement. The generic report's interpolated EER differs numerically from original BEACON's nearest-observed-point EER even for the same LR; documented prominently in `beacon_nonlinear/README.md` to prevent a spurious improvement claim. Existing-dev exposure remains explicit in the protocol and result.

## Nonlinear serving provenance hardening

Independent integration review verified exported schema thresholds equal corrected `frozen_v2.json`, but identified that the serving loader previously trusted local schema/joblib without rechecking source provenance. Added a nonlinear-only immutable eleven-file export integrity manifest with its SHA256 pinned in router code. Uncached schema/model loading now verifies schema, four models, corrected freeze, protocol, executed snapshot and original BEACON source/configuration/split hashes. Failures return HTTP 503 before deserialization; joblib receives the verified bytes rather than reopening a file.

Cached loaded objects are not continuously rechecked, explicitly documented. Thirteen affected tests plus five tamper subcases pass, including tamper rejection after schema caching and corrected-threshold synthetic ASGI checks. No fitted artifacts or reports were modified, no measurements were used, and no dev replay or network listener was started for this hardening change.
