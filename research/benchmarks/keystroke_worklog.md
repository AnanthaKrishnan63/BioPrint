---
title: Keystroke acquisition and experiment worklog
tags: [research, worklog, experiments]
updated: 2026-09-20
---

# Keystroke worklog

## 2026-09-20 — acquisition and protocol

- Read repository guidelines and the existing CMU evaluator source without running it. Confirmed it parses all timing rows and uses historical held-out sessions for published comparisons; legacy results cannot support pristine-holdout claims.
- Researched primary KeyRecs, Type2Branch, TypeFormer, and 2026 lightweight KeyRecs sources. Chose the 31 MB KeyRecs download because both relevant tracks fit the hardware/dataset budgets. Aalto mobile's documented 5.6 GB archive does not.
- Downloaded official Zenodo metadata and two CSVs, verified published MD5 hashes. Source license is CC-BY-4.0. No demographic table downloaded.
- Established deterministic metadata-only identity partition, sealed 20 subjects, left 79 active. Saved immutable split manifest before parsing training timings. No sealed measurements inspected.
- Implemented bounded acquisition, hash verification, manifest protection, and a train/dev-only adapter. Implemented CPU LR, shrinkage LDA, and ExtraTrees evaluation with chronological training-only fit/select/calibration blocks and frozen session-2 validation.
- First execution stopped at import because repository-local sklearn required `narwhals`. No fitting or validation occurred. Root agent installed the missing dependency. Restarted only after the process was confirmed exited.

## Experiment K1 — preregistered configuration

- Fixed inputs: all 47 numeric timing columns, preserving repeated column names by position; no subject/session/repetition features.
- Free inputs: 50-digraph disjoint chronological windows, five timing channels, quantiles 10/25/50/75/90%, mean and SD (35 features). No key identities or content features.
- Training blocks per subject: first 50% fit, next 25% select, final 25% calibrate. No overlap. Training-fitted imputation, 0.5/99.5 percentile clipping, standardization.
- Candidate grid set before dev access: LR C=0.1/1/10; LDA shrinkage=0.1/0.5; ExtraTrees 128 trees, minimum leaf size=2/5. Select minimum mean per-subject EER on training selection block.
- Threshold: per-account empirical FAR <=1% on training calibration impostors, ties conservatively rejected. Freeze before dev loading.
- Report both training-selected LR reference and training-selected overall method on dev, including FRR, FAR, EER and labeled oracle FRR@1% FAR. This is an adapted recent-method baseline, not a SOTA reproduction.
- Execution log: `keyrecs-run-v1.log`. Frozen artifacts/results: `results/keyrecs-v1/`. Results will be recorded after completion.

## Experiment K1 — completed validation

| Track / frozen method | Dev EER | FRR at frozen threshold | Achieved FAR |
|---|---:|---:|---:|
| Fixed / LR C=0.1 | 17.080% | 76.389% | 1.257% |
| Fixed / ExtraTrees leaf=2 | 11.618% | 45.914% | 1.293% |
| Free / LR C=0.1 | 16.665% | 85.726% | 0.705% |
| Free / ExtraTrees leaf=5 | 9.723% | 45.644% | 1.141% |

- Every operating threshold was fixed at training-calibration FAR <=1%; achieved dev FAR is reported separately, not mislabeled exactly 1%. User bootstrap intervals and per-user rows are in JSON reports. Training selection chose both ExtraTrees models before dev loading. No improvement was selected in response to dev results.
- Fixed validation: 79 users, 7,896 repetitions. Free validation: 79 enrolled users but 78 evaluable users, 4,525 nonoverlapping windows; p004 has no complete 50-digraph dev window and is disclosed rather than silently included in macro metrics.
- The first free loader attempt encountered 18 training rows with literal quote key labels that violate normal CSV quoting. The timing suffix remained intact. Adapter now parses the five numeric suffix fields from the right only after checking allowed identity/session; key text is omitted. No malformed test rows were inspected.
- Free validation initially stopped on an empty-window array. Fixed its shape to `(0, 35)` and added `--validate-frozen`, then resumed from the saved models/thresholds without fitting again. Fixed-model joblib files were repacked to use importable class paths, without changing coefficients or scores.
- Synthetic checks pass: a poisoned sealed row never reaches numeric parsing, test reads fail, source tampering fails, quote-key rows parse, score ties respect FAR constraints, and chronological training blocks do not overlap. Command: `python scripts/keystroke_benchmark.py --self-check` in bigidea.

## Research API integration

- Added opt-in `code/bioprint/research_api.py`, separate from the production app and DB. It rejects nonloopback clients, foreign Host headers and foreign origins, exposes train/dev pages, refuses test paths, and scores with frozen KeyRecs/CMU models. CMU distance sign is inverted so all research scores use larger-is-genuine; frozen thresholds are inverted with it.
- Added `scripts/api_dataset_replay.py` to send every dev feature vector through API endpoints, compare scores with offline artifacts, and recompute metrics using frozen thresholds. It never starts a server by default; `--in-process` uses ASGI transport and the normal middleware/routes.
- Sandbox runs hung in the AnyIO threadpool. A five-second diagnostic stack confirmed the environment issue, and the exact two stuck verification Python processes were terminated. Escalated isolated ASGI tests then completed: **8 passed in 0.99s**. No LAN server started and no production DB imported.
- Full ASGI replay completed: 7,896 fixed +4,525 free +5,100 CMU dev samples. Maximum score discrepancies were 1.1e-16, 2.2e-16, and 3.4e-13 respectively; frozen decisions and achieved operating rates matched. Fixed diagnostic EER differs by <0.0001 percentage point because machine-precision score ties can reorder; this is numerical noise, not improvement. Report: `results/keyrecs-v1/api-replay.json`.

## Experiment T1 — TypeNet architecture transfer preregistration

- Investigated official Type2Branch tree and releases metadata. No pretrained checkpoint in either; source HOWTO offers training instructions only. Official TypeNet embeddings require a signed emailed license and would not themselves provide a model. No permission emails were sent.
- Implemented the published TypeNet architecture in PyTorch, including actual locked recurrent dropout rather than substituting ordinary inter-layer dropout. Synthetic checks confirm exactly 200,458 parameters, 128D embeddings, deterministic evaluation, stochastic training, finite gradients and sealed-test rejection.
- Predeclared CPU schedule and data adaptations live in `results/typenet-keyrecs-v1/preregistered.json`; training history and source snapshot are saved with artifacts. Hyperparameter selection/early stopping uses the chronological S1 internal selection block only; session2 is opened only after freezing the model, gallery and calibration thresholds.
- T1 starts from random initialization because authentic pretrained weights were not found in the official distributions. It cannot establish open-set or full-scale published SOTA performance. Raw data and the K1 results remain unchanged.

## Experiment T1 — completed validation

- The preprocessing preflight found missing numeric values in training rows. Applied the already declared fit-only missing-value imputation instead of raising on empty strings; reran the identical preregistered configuration before any optimization step had completed. No model or dev evaluation was repeated.
- Source learning rate 0.05 stopped after 15 epochs under the declared patience rule. The predeclared 0.001 adaptation completed 20 epochs, selected epoch20 with **18.460% training-selection EER**. Fitting used 2,194 sequences; selection 1,096; calibration 1,134. Training took 677.1 seconds on two CPU threads. Saved checkpoint is 807,879 bytes.
- Training source audit: 89 unknown/malformed keycodes mapped to zero, including 18 malformed literal-quote prefixes; 1,878 tail digraphs discarded because only full nonoverlapping 50-event windows were admitted. These counts came only from training.
- Frozen dev result: **EER20.233%, FRR84.084%, achieved FAR1.172%** at a threshold calibrated to training FAR<=1%. EER subject-bootstrap95% interval:18.05–22.81%. Dev cohort/window count matches the free-text K1 track:78 evaluable users,4,525 windows.
- This neural architecture transfer **underperformed** the train-selected ExtraTrees baseline. It is retained as a negative result and evidence of an actual modern sequence-method experiment, not promoted or described as reproduced SOTA. No post-dev model/threshold tuning occurred.
- Custom LSTM recurrence agrees with PyTorch's reference recurrence when recurrent dropout is disabled (maximum synthetic discrepancy5.96e-8). The check is now included in the script's `--stage self-check` path.
- TypeNet250-feature inference was added to the isolated research API. A regression test protects it from accidental substitution with the35-feature summary representation. Security/schema suite: **9 passed in1.33s**.

## Paired neural transfer and end-to-end replay

- Full four-track API replay now includes TypeNet:22,046 development samples; every frozen threshold decision matches offline inference. Maximum TypeNet score discrepancy1.78e-15. Report:`results/keyrecs-v1/api-replay-v2.json`.
- Implemented frozen KeyRecs TypeNet transfer into original BEACON30second paired windows. Authoritative support ledger binds enrollment to training roles, including personal support for dev identities; global fusion fits only P002/P003/P006, selects P007/P008, calibrates P009/P010. DevP012/P015/P022 is validation only; testP026/P028 remains sealed.
- Mapping audit validates recorder elapsed seconds against Duration, rejects unknown keycodes>1%, and keeps source logger auto-repeat and gameplay-domain limitations explicit. Short keyboard sequences gather last valid recurrent state, tested against an unpadded forward pass.
- V1 neural/handcrafted arrays preserve all original243 development comparisons and the exact baseline mouse/key features. Diagnostic EER TypeNet37.04%, mouse40.43%, neural+mouse39.51%, hybrid43.21%. Training FAR1% calibration transfers poorly: TypeNet rejects all genuine probes; neural+mouse achieves FAR41.98%/FRR34.57%. These results do not justify deployment or a claimed improvement.
- New isolated neural API replays all four models across243 comparisons, exact score/decision parity, test403 and nonloopback403. Report:`beacon-neural-v1/api_replay.json`. No production imports, database writes or listener.
- Preregistered V2 frozen SapiMouse FCN + TypeNet transfer in`beacon-neural-v2/declaration.json` before its fitting/validation. FCN encoder hash pinned. Same original30second paired windows, deterministic requirement>=129 raw pointer events, no padding/cross-window joins; all six comparator models share identical eligible records and train-only candidate grid.
- V2 complete:377/378 train and140/141 dev support+probe windows eligible, no identity exclusion.243 dev comparison labels and claimed/actual/window groups exactly matchV1. Handcrafted features can differ because eligible enrollment support changed. Pairing audit saved.
- V2 dev diagnostic EER:TypeNet35.80%, handcrafted42.90%, FCN59.26%, dual neural59.26%, hybrid43.21%. At frozen training1%FAR threshold, dual achieves FAR54.94%/FRR59.26%; hybrid FAR4.32%/FRR92.59%; TypeNet rejects all genuine probes. Negative source transfer retained without post-dev tuning.
- V2 isolated API replay:all six models ×243 comparisons have exact scores and every frozen operating-point decision agrees with offline results; test/nonloopback403. Neural routes converted to async to avoid environment threadpool failure. Finite-but-unrepresentable values such as1e308 receive422 before model loading. Synthetic regression/security suite10passed1.03s.

## Runtime-only optimization audit

- Generated-vector alternating timing verifies ExtraTrees worker-startup overhead: fixed singlequery median59.75ms with2threads versus14.92ms with1; free59.17ms versus14.13ms.128-row batch timings also improve. These measurements ran alongside other research processes and are indicative, not a controlled hardware benchmark.
- All981,259 development identity scores from12,421 fixed/free samples are bit-exact to frozen artifacts after setting the loaded estimator's `n_jobs=1`. Every frozen accept/reject flag agrees, including the closest fixed-text score only1.39e-17 from its threshold. No model weights, thresholds or serialized artifacts changed.
- Enabled single-thread scheduling only inside the isolated research API loader. Source training configuration remains unchanged. Runnable audit:`scripts/keyrecs_runtime_audit.py`; report:`results/keyrecs-v1/runtime-audit.json`. Selection uses generated timings; development parity can only veto the runtime change, never tune identity statistics.
- Complete actual ASGI integration replay also passed after the loader change:7,896 fixed+4,525 free samples, exact scores and all frozen decisions unchanged, sealed-test403. Report:`results/keyrecs-v1/api-replay-serial.json`. Homepage now links both paired neural reports.

## Bounded Type2Branch feasibility follow-up

- Read additional pinned author HOWTO, loss, generator, training, preprocessing and external-feature merger sources. Audited small configuration has150-sequence batches, two256-unit bidirectional GRUs and128/256/512 CNN branch; true Set2Set is not ordinary triplet loss. Maximum schedule60,000 updates plus validation/centroid work.
- Reference TensorFlow/Keras/Addons dependencies are absent from Python3.12 bigidea; documented Addons compatibility/wheels stop at3.11. Stock source protocol reserves1,000 validation and1,000 evaluation users; existing KeyRecs has99 total. Extended-feature merger requires externally generated CSVs whose producer is absent from the17-file repository.
- Decision:do not present a hurried unvalidated port as faithful reproduction. Saved concrete source contracts, dependency/data gaps, required equivalence checks and future epoch benchmark in`type2branch_feasibility.md`. No new dev evaluation, model modification, gated access or training run. Existing TypeNet is explicitly retained as the implemented architecture experiment; Type2Branch remains an unmet modern-SOTA reproduction item.

## Bot-rule actual API coverage

- Added isolated`research_bot_api.py` routes for paginated genuine CMU dev reconstructed Samples with owner/session/repetition metadata, bounded128-row inference, and results with scope caveats. Each claim uses only that owner's first10 session1 training vectors; probe history never updates enrollment. Sealed splits reject before loading.
- Frozen bot source hash initially rejected root's later wording edits. Verified gitHEAD exactly matches original frozen hash and current AST differs only in docstrings/comments and the third explanation argument of`acc.add("impossible_hold",...)`. Recorded restricted compatibility audit with both exact hashes; all other source gates remain enforced and frozenplan unchanged.
- `scripts/bot_api_replay.py` replayed all5,100 actual endpoint samples:exact score, flag, ordered rule-list and aggregate/per-rule parity against stored offline observations;15 genuine samples flagged. Test requests403. No invented browser environment/pointer telemetry, listening server or database access. Report:`bot_rules/api_replay.json`.
- Results explicitly do not validate browser event trust (assumedtrue), real browser environment or bot-positive sensitivity/FAR/EER. This is legacy CMU timing-rule genuine false-flag validation only.

## CMU account-specific training selection

- Inspected only existing strict-loader source and TRAIN-selection artifact for feasibility. Original19-candidate run took68.5seconds. A predeclared nine-SVM candidate follow-up tests account-specific C/gamma choice using the same grid, exact10sample session1 enrollment, sessions2–3 selection, session4 threshold calibration, and unchanged background/features.
- Wrote`cmu-account-selection/preregistered.json` before any measurements for the new run, pinning original artifacts and source. Script`cmu_account_selection.py` has no dev evaluation entrypoint. Exact ties use fixed original grid order. No grid extension or refitting after selection.
- Prior exposure to global dev results and the legacy corpus is explicit. Minimizing each account's loss on the existing selection set necessarily produces optimistic aggregate training-selection gains; this does not establish generalization. Baseline/global frozen models remain unchanged and are copied as matched-enrollment controls.
- Training completed in26.42seconds onCPU2. Macro selection EER9.2094% versus global11.8573%. All51 selected account models and three session4 thresholds are frozen. All nine grid choices occur; full account/candidate losses and choice counts are saved. Verified copied control models are exactly unchanged and recomputed global selection loss equals the original artifact. Stopped before dev validation and reported artifacts to parent.
- Parent then authorized one frozen dev comparison. Wrote a separate source/hash-pinned validation plan before reading sessions5–6; script`cmu_account_validate.py` has no fitting or model-selection path and refuses overwrite. Saved originalCMU-shaped`dev_results.json` plus`dev_scores.npz` containing all three5100×51 distance matrices, subjects and labels.
- Dev EER14.3245% versus global15.5024%. At trainingFAR1 threshold:achieved FAR0.9486%/FRR67.4314%, versus global0.9980%/72.9804%. At trainingFAR5 threshold:4.8718%/38.0000%, versus global4.9039%/44.7647%.
- Paired-account bootstrap delta versus global: EER−1.178percentagepoints,95%interval[−2.560,+0.257], so no established EER improvement. FRR delta at trainingFAR1−5.549points,[−10.490,−1.430]; at trainingFAR5−6.765points,[−10.726,−3.509]. These exploratory intervals resample accounts and do not fully model shared-impostor dependencies. No post-dev changes were made. Parent/otheragent own actual API integration/replay.

## CMU selection aligned with low-FAR objective

- No duplicate experiment found. Preregistered`cmu-far-selection/preregistered.json` before measurement reads: same nine SVM candidates, first10 session1 samples, unchanged28features/background. Per account select lowest FRR at empirical1%FAR on sessions2–3; tie by EER then fixedgrid. Session4 solely final threshold calibration; selection thresholds never deployed. Hypothesis is the mismatch between EER optimization and the low-FAR operational objective. Prior development exposure is explicit.
- TRAIN-only run completed25.29seconds CPU2. SelectionFRR at1%FAR:45.0980% versus EER-account51.1569%, global63.8431%, baseline63.6863%. SelectionEER trades off to11.7669% versus EER-account9.2094%. Choices changed28/51accounts; fullcandidate metrics/choicecounts stored. These reused-selection gains are optimistic and do not establish generalization.
- Gate passed because selectionFRR improved6.0588percentagepoints and models are nonredundant. Frozen artifact contains baseline/global/EER-account controls copied exactly unchanged plus51 calibrated`far_selected` models. Stopped before newdev/test reads and reported to parent. Two synthetic tests cover objective precedence and deterministic tie resolution;2passed0.40s.
- Parent authorized one separate frozen validation. `cmu_far_validate.py` wrote source/artifact/hash-pinned plan before sessions5–6; four models compared with identical2,000 account-bootstrap resamples and no retuning. EER18.50%, FAR/FRR0.99%/64.10% at training1% threshold,4.93%/39.92% at training5% threshold.
- Versus EER-account selection, FRR at training1% falls3.33percentagepoints (paired95%interval−6.96 to−0.37), with FARchange+0.037points (−0.030 to+0.099). EER worsens4.17points (+1.43 to+7.56); training5%FRRchange+1.92points (−1.22 to+5.24). This is an operating-point tradeoff, not universal improvement. Prior exposure, exploratory multiplicity and shared-impostor dependence limit inference. OwnREADME and full score matrices saved; parent/pointeragent ownAPIreplay.
