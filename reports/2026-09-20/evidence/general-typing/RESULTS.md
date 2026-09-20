# General typing matcher: measured results

This experiment removes the CMU password-schema restriction. It does not establish universal password, language, keyboard, mobile, or attacker generalization. The four login experiment branches retain their existing scorers; this work is isolated in `experiment/general-typing`.

## Method

A shared binary matcher compares an attempt with ten enrollment recordings. For each of hold, down-down and up-down timing, calculate five quantiles, mean and standard deviation. This gives 21 features for any supported password length. For feature j, c_j = median(enrollment_j), s_j = max(mean absolute deviation, 10 ms, 0.1 |c_j|), z_j = clip((attempt_j - c_j)/s_j, -100, 100). The learned models receive [|z|, z], 42 features, without key labels or identity IDs. The distance comparator uses -mean(min(|z|, 6)). Larger scores indicate a closer match.

Fit, threshold-calibration and evaluation identities are disjoint. KeyRecs identities retain the same role across fixed/free tracks. Each evaluation identity contributes only its first ten valid TRAIN records to personal enrollment; its DEV records never fit the population model or threshold. All eligible observations are used, with invalid vectors and free-window tails counted below. Sealed test partitions remain closed.

Logistic regression uses C=1; ExtraTrees uses 128 trees and minimum leaf size 10. Fit weights balance track and genuine/impostor class. All nine configurations were frozen before DEV reads; there was no candidate search or DEV-based retuning. The threshold is just above the relevant calibration impostor order statistic, so at most floor(0.01 N_impostor) calibration impostors pass. This is an empirical calibration constraint, not a guarantee on DEV FAR.

FAR = false accepts / impostor comparisons; FRR = false rejects / genuine comparisons. EER is the mean of FAR and FRR at the empirical ROC point with the smallest difference (diagnostic only). Tables use pooled comparisons, not the previous reports’ macro-user EER, so compare only rows here. All-pairs comparisons share people and probes; they are not independent trials.

## Pooled training: unseen evaluation users

| Track | Method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |
|---|---|---:|---:|---:|---:|---:|
| cmu | Existing positional scorer, enrollment threshold | 33.91% | 18.80% | 26.31% | 3052/9000 | 188/1000 |
| cmu | distance | 0.21% | 99.90% | 38.02% | 19/9000 | 999/1000 |
| cmu | logistic | 0.87% | 99.20% | 28.32% | 78/9000 | 992/1000 |
| cmu | extra_trees | 1.62% | 98.80% | 28.00% | 146/9000 | 988/1000 |
| fixed | Existing positional scorer, enrollment threshold | 50.97% | 13.46% | 31.64% | 9257/18161 | 188/1397 |
| fixed | distance | 0.11% | 99.86% | 35.08% | 20/18161 | 1395/1397 |
| fixed | logistic | 0.19% | 99.57% | 37.72% | 34/18161 | 1391/1397 |
| fixed | extra_trees | 0.36% | 96.35% | 34.86% | 65/18161 | 1346/1397 |
| free | distance | 0.83% | 91.79% | 30.02% | 493/59553 | 4205/4581 |
| free | logistic | 1.90% | 93.25% | 31.61% | 1129/59553 | 4272/4581 |
| free | extra_trees | 1.57% | 90.68% | 28.60% | 936/59553 | 4154/4581 |

The existing scorer comparison uses identical eligible people and attempts. Its enrollment-derived threshold is its shipped operating point; the general models instead use a global calibration-FAR target. EER helps separate score discrimination from operating-point differences. Free text has no fixed positional password template, so an existing-login baseline there would be misleading.

## Cross-dataset transfer

These models and thresholds see only the named source dataset. Evaluation users in the target are unseen; their personal enrollment is still required. CMU and KeyRecs fixed use different prompted texts, but this is only a small number of passwords and cannot establish accuracy for arbitrary secrets.

| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |
|---|---|---:|---:|---:|---:|---:|
| cmu | keyrecs_only / distance | 0.17% | 99.90% | 38.02% | 15/9000 | 999/1000 |
| cmu | keyrecs_only / logistic | 0.57% | 99.40% | 28.40% | 51/9000 | 994/1000 |
| cmu | keyrecs_only / extra_trees | 0.46% | 99.50% | 32.10% | 41/9000 | 995/1000 |
| fixed | cmu_only / distance | 0.70% | 99.21% | 35.08% | 127/18161 | 1386/1397 |
| fixed | cmu_only / logistic | 2.33% | 93.41% | 34.16% | 424/18161 | 1305/1397 |
| fixed | cmu_only / extra_trees | 0.89% | 91.77% | 33.07% | 161/18161 | 1282/1397 |
| free | cmu_only / distance | 5.14% | 71.97% | 30.02% | 3061/59553 | 3297/4581 |
| free | cmu_only / logistic | 1.02% | 94.04% | 31.85% | 607/59553 | 4308/4581 |
| free | cmu_only / extra_trees | 2.97% | 94.22% | 33.66% | 1769/59553 | 4316/4581 |

## Single-source within-dataset controls

| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |
|---|---|---:|---:|---:|---:|---:|
| cmu | cmu_only / distance | 1.57% | 98.40% | 38.02% | 141/9000 | 984/1000 |
| cmu | cmu_only / logistic | 0.79% | 99.20% | 26.40% | 71/9000 | 992/1000 |
| cmu | cmu_only / extra_trees | 3.12% | 94.90% | 26.91% | 281/9000 | 949/1000 |
| fixed | keyrecs_only / distance | 0.09% | 99.86% | 35.08% | 17/18161 | 1395/1397 |
| fixed | keyrecs_only / logistic | 0.24% | 99.36% | 36.93% | 44/18161 | 1388/1397 |
| fixed | keyrecs_only / extra_trees | 0.14% | 99.79% | 37.74% | 25/18161 | 1394/1397 |
| free | keyrecs_only / distance | 0.67% | 93.21% | 30.02% | 401/59553 | 4270/4581 |
| free | keyrecs_only / logistic | 1.58% | 94.08% | 31.98% | 943/59553 | 4310/4581 |
| free | keyrecs_only / extra_trees | 1.24% | 90.42% | 27.02% | 738/59553 | 4142/4581 |

## BEACON: frozen gameplay transfer

All 3 existing DEV identities were eligible. First ten disjoint ten-event windows from each TRAIN-enrollment recording form the personal profile. Every valid probe window is compared against every profile. No BEACON data fits the population model or threshold. See BEACON_PROTOCOL.md and results/beacon.json for source guards and exclusion counts.

| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |
|---|---|---:|---:|---:|---:|---:|
| BEACON gameplay | pooled/distance | 9.97% | 91.80% | 51.43% | 73/732 | 336/366 |
| BEACON gameplay | pooled/logistic | 7.51% | 93.72% | 54.92% | 55/732 | 343/366 |
| BEACON gameplay | pooled/extra_trees | 4.64% | 96.99% | 53.89% | 34/732 | 355/366 |
| BEACON gameplay | cmu_only/distance | 22.40% | 86.34% | 51.43% | 164/732 | 316/366 |
| BEACON gameplay | cmu_only/logistic | 5.19% | 96.99% | 48.91% | 38/732 | 355/366 |
| BEACON gameplay | cmu_only/extra_trees | 0.96% | 98.63% | 48.91% | 7/732 | 361/366 |
| BEACON gameplay | keyrecs_only/distance | 8.47% | 92.62% | 51.43% | 62/732 | 339/366 |
| BEACON gameplay | keyrecs_only/logistic | 5.19% | 93.99% | 55.19% | 38/732 | 344/366 |
| BEACON gameplay | keyrecs_only/extra_trees | 5.33% | 97.27% | 52.53% | 39/732 | 356/366 |

## Data and exclusions

| Track | Fit identities | Calibration identities | Evaluation identities |
|---|---:|---:|---:|
| cmu | 28 | 13 | 10 |
| fixed | 47 | 18 | 14 |
| free | 47 | 18 | 14 |

| Partition | Valid vectors | Excluded vectors | Unused tail digraphs |
|---|---:|---:|---:|
| cmu/train | 10200 | 0 | 0 |
| fixed/train | 7878 | 0 | 0 |
| free/train | 24745 | 9 | 292 |
| cmu/dev | 5100 | 0 | 0 |
| fixed/dev | 7896 | 0 | 0 |
| free/dev | 25306 | 9 | 298 |

Total fitting and validation runtime: **784.5 seconds**.

## Verification and inference cost

Independent audit checked 36 saved score tables, 9 artifact hashes, exact operating counts, source hashes, identity separation and calibration FAR constraints. Fifteen unit tests cover arbitrary key sequences, variable lengths, live/dataset timing parity, invalid timings and browser-format input. Generated input latency includes feature extraction and enrollment-profile construction, excludes model loading, and is indicative on a machine also running research jobs.

| Pooled method | Median inference (ms) | Maximum of 10 calls (ms) | Artifact size (MiB) |
|---|---:|---:|---:|
| distance | 10.68 | 11.39 | 0.000 |
| logistic | 11.75 | 12.47 | 0.003 |
| extra_trees | 85.82 | 90.82 | 534.007 |

## Limits and integration decision

This is offline typing verification, not all-feature app FAR/FRR/EER. Free-text windows contain nine source digraphs and are a short-input stress test; different texts can contribute to apparent identity separation. Some KeyRecs down-down timings are negative because source event ordering differs from browser press ordering. Distribution pooling loses exact key-position information. Prior DEV exposure remains disclosed. No automatic profile updates are enabled.

No candidate is automatically promoted by this report. A login integration needs a predeclared calibration gate, browser timing parity, latency checks and tests of people entering the same arbitrary password. A new password can be represented by this model; that does not mean its error rates have been measured.

## Reproduction and artifacts

From the original repository root, using the bigidea environment:

```bash
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/run.py
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/control.py
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/beacon.py
bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/audit.py
bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/report.py
```

`run.py` refuses an existing results directory. Preserve the current run before reproducing. The guarded loaders and bigidea launcher are existing research workspace dependencies. `results/plan.json` pins input/source hashes and identity roles; `frozen.json` pins trained artifacts; `report.json` includes every evaluation and per-user statistics; local `*-scores.npz` files retain all score/label pairs. Large joblib models and score arrays remain local and are reproducible, not Git blobs.

`infer.py --input recording.json --model pooled/logistic` accepts ten enrollment Sample objects and an attempt in browser format. It verifies model checksums and returns only a behavioral score; it does not authorize login or start a server.
