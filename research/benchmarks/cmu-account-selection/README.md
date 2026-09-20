# Per-account CMU SVM selection

This exploratory follow-up selects SVM regularization and kernel width separately
for each account. It preserves the original ten-sample enrollment budget, timing
features and candidate grid. It is an adapted baseline, not a SOTA reproduction.

## Partition and selection contract

- Fit: first ten repetitions of session 1 for each claimant; unchanged other-user
  negative examples and population scaling.
- Selection: sessions 2–3, minimizing account EER; fixed candidate order breaks ties.
- Calibration: session 4 only, with frozen EER, 1% FAR and 5% FAR thresholds.
- Dev: sessions 5–6, evaluated after freezing; sessions 7–8 remain sealed.

Previous CMU exploration and earlier dev reports were already exposed. These
results cannot serve as independent confirmation. Model choices and thresholds
were not adjusted after this dev run.

## Frozen results

| Method | Macro dev EER | Actual FAR at training 1% target | FRR |
|---|---:|---:|---:|
| Original baseline | 22.04% | 1.10% | 74.20% |
| Global SVM setting | 15.50% | 1.00% | 72.98% |
| Account-specific settings | 14.32% | 0.95% | 67.43% |

The paired-account EER difference interval includes zero. The FRR difference
is −5.55 percentage points, with a paired-account bootstrap interval of
−10.49 to −1.43. Shared impostor observations limit that bootstrap interpretation.
Substantial false rejection remains; the model is not enabled for production users.

## Artifacts and scripts

`preregistered.json` and `validation_plan.json` precede their corresponding data
reads. `frozen_models.json` includes unchanged baseline/global controls and every
account's selected model. `dev_results.json` reports metrics and uncertainty;
`dev_scores.npz` contains three distance matrices, labels and subject order.

Run scripts from the repository root with `bash scripts/research.sh`:
`cmu_account_selection.py`, `cmu_account_validate.py` and
`cmu_account_api_replay.py` are in `scripts/`. Inspect their command-line interfaces
and existing artifacts first; completed outputs are protected against overwrite.
`source.py`, `validation_source.py` and `api_replay_source.py` preserve executed
versions. API replay verifies 780,300 scores with zero decision flips. Provenance
notes distinguish that replay from subsequent malformed-input regression tests.
