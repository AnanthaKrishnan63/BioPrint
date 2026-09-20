# General matcher with personal password alignment

This is the second, exploratory experiment on branch `experiment/general-typing`. It was declared after the summary-only model failed on DEV, so reused DEV cannot provide independent confirmation. No server or login policy was changed.

## How it works for different passwords

Each user still enrolls their own password ten times. The existing profile supplies a center c_j and spread s_j for each key-position timing. For a new attempt, compute z_j = clip((x_j-c_j)/s_j, -6, 6). Summarize these residuals separately for H, DD and UD timings: five absolute quantiles, absolute mean/standard deviation and signed median/mean/standard deviation. Append the existing distance/threshold ratio. These 31 features have the same meaning and size for different passwords. The shared logistic/tree model sees neither key labels nor account IDs. Alignment remains local to the account; unrelated passwords are never aligned.

A new password therefore does not require retraining the shared model, but does require a new personal enrollment profile. This is not automatic online adaptation.

## Evaluation

Same guarded CMU and KeyRecs-fixed data, identity roles and trials as RESULTS.md. Fit, calibration and evaluation identities are disjoint. All nine model/threshold combinations were frozen before this follow-up read DEV. Thresholds target empirical calibration FAR<=1%; DEV FAR can differ. No hyperparameter search or threshold retuning. The distance comparator is also recalibrated, so gains cannot be credited merely to tightening the shipped threshold. Free text and gameplay are excluded here because they do not repeat the enrolled password sequence.

FAR=FA/N_impostor, FRR=FR/N_genuine. EER is a diagnostic pooled empirical ROC statistic, not a deployable threshold. Comparisons share users and are dependent. This cohort has ten CMU and fourteen KeyRecs evaluation identities; these numbers must not be substituted into the earlier full-cohort reports.

| Dataset | Training / method | FAR | FRR | EER | FA / impostors | FR / genuine | EER change vs shipped scorer (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| cmu | Shipped positional scorer | 33.91% | 18.80% | 26.31% | 3052/9000 | 188/1000 | +0.00 |
| cmu | pooled/distance | 1.44% | 97.20% | 26.31% | 130/9000 | 972/1000 | +0.00 |
| cmu | pooled/logistic | 0.42% | 95.30% | 25.41% | 38/9000 | 953/1000 | -0.90 |
| cmu | pooled/extra_trees | 1.72% | 90.90% | 24.19% | 155/9000 | 909/1000 | -2.12 |
| cmu | cmu_only/distance | 4.14% | 90.60% | 26.31% | 373/9000 | 906/1000 | +0.00 |
| cmu | cmu_only/logistic | 4.12% | 75.70% | 25.62% | 371/9000 | 757/1000 | -0.69 |
| cmu | cmu_only/extra_trees | 5.91% | 71.00% | 24.30% | 532/9000 | 710/1000 | -2.01 |
| cmu | keyrecs_only/distance | 0.88% | 98.80% | 26.31% | 79/9000 | 988/1000 | +0.00 |
| cmu | keyrecs_only/logistic | 0.66% | 95.40% | 28.98% | 59/9000 | 954/1000 | +2.67 |
| cmu | keyrecs_only/extra_trees | 0.87% | 98.40% | 25.92% | 78/9000 | 984/1000 | -0.39 |
| fixed | Shipped positional scorer | 50.97% | 13.46% | 31.64% | 9257/18161 | 188/1397 | +0.00 |
| fixed | pooled/distance | 3.56% | 87.47% | 31.64% | 646/18161 | 1222/1397 | +0.00 |
| fixed | pooled/logistic | 3.83% | 88.98% | 29.42% | 695/18161 | 1243/1397 | -2.22 |
| fixed | pooled/extra_trees | 4.58% | 82.61% | 30.78% | 831/18161 | 1154/1397 | -0.86 |
| fixed | cmu_only/distance | 7.41% | 75.02% | 31.64% | 1346/18161 | 1048/1397 | +0.00 |
| fixed | cmu_only/logistic | 9.79% | 68.58% | 30.57% | 1778/18161 | 958/1397 | -1.07 |
| fixed | cmu_only/extra_trees | 9.60% | 67.93% | 29.92% | 1744/18161 | 949/1397 | -1.72 |
| fixed | keyrecs_only/distance | 2.42% | 90.98% | 31.64% | 440/18161 | 1271/1397 | +0.00 |
| fixed | keyrecs_only/logistic | 4.25% | 90.62% | 30.62% | 772/18161 | 1266/1397 | -1.02 |
| fixed | keyrecs_only/extra_trees | 2.94% | 90.05% | 32.14% | 534/18161 | 1258/1397 | +0.50 |

Negative EER change means improvement. `cmu_only` on KeyRecs and `keyrecs_only` on CMU are cross-dataset/password transfer. All other rows are unseen-user evaluation with the dataset represented in population fitting. Only two fixed password texts are represented, limiting generalization claims.

## Verification and runtime

Fitting and evaluation took 162.7 seconds. The audit verified 18 score tables, nine model hashes, source hashes, identity separation and identical trial counts against the shipped scorer. Fifteen tests cover the two representations and browser adapters. Median generated-input inference latency (including personal profile construction, excluding artifact load): distance 5.24 ms, logistic 6.90 ms, extra_trees 64.99 ms.

## Reproduce and use offline

```bash
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/aligned.py
bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/finish_aligned.py
bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/infer.py --representation aligned --model pooled/logistic --input recording.json
```

The run refuses an existing aligned_results directory. Keep original evidence. Input contains ten browser-format enrollment Sample objects and one attempt. The command returns a behavioral score, not authentication authorization. Credentials, bot checks and device context remain separate.

All model binaries and score arrays are available locally; hashes and reports are committed. No candidate is deployed automatically. These are single-attempt typing results, not three-repetition, pointer-step-up or all-feature app error rates.
