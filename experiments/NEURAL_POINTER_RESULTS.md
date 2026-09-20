# Trained pointer integration: synthetic FAR / FRR

The trained SapiMouse FCN + cosine model is now integrated into the desktop `typing-pointer` step-up path after personal pointer enrollment. This report measures a synthetic cross-dataset composition through the actual API and encoder, not real people using the combined app.

## Paired comparison

| Configuration | False accepts | FAR | False rejects | FRR |
|---|---:|---:|---:|---:|
| Previous pointer motor scorer | 14/90 | 15.56% | 13/60 | 21.67% |
| Trained pointer, current threshold | 9/90 | 10.00% | 19/60 | 31.67% |

FAR improves by 5.56 percentage points; FRR worsens by 10.00 points. This is five fewer false accepts and six additional false rejects, not an across-the-board improvement.

Both policies require additional verification in 63/150 cases. The new encoder decides 33/150 desktop cases; mobile retains keypad verification. Incomplete checks: 0. The replay reproduced every stored legacy decision and checked session-cookie outcomes after all 300 paired cases. All 33 neural cosine scores matched their frozen research predictions within 2e-6.

## FAR versus FRR at existing TRAIN-calibrated thresholds

These are the three thresholds already frozen by the pointer research. Only the 1% source-calibration target is currently used in enrollment. Other rows are diagnostic score-level counterfactuals on the same recorded attempts; no threshold was retuned or promoted.

| Source calibration FAR target | Cosine threshold | App synthetic FA / impostors | FAR | FR / genuine | FRR |
|---|---:|---:|---:|---:|---:|
| 0.1% | 0.956120253 | 9/90 | 10.00% | 20/60 | 33.33% |
| 1% | 0.948082149 | 9/90 | 10.00% | 19/60 | 31.67% |
| 5% | 0.935335815 | 9/90 | 10.00% | 17/60 | 28.33% |

The source calibration target is not the combined app FAR. The branching policy has no prespecified single scalar score, so an all-feature EER is not reported.

## Why false accepts remain

All nine remaining false accepts bypass the neural challenge: seven typing-match attacks pass direct admission, and two mobile impostor cases pass the unchanged keypad. Changing a desktop step-up scorer cannot catch an attempt that never reaches it. The next policy question is whether direct admission needs another independent signal, balanced against extra friction. This report does not change that policy.

## Synthetic construction and limits

The original 15 synthetic accounts and 150 DEV cases are unchanged. Their CMU typing is combined with separately enrolled SapiMouse users by a frozen seeded mapping. Real people are not linked across datasets. The run preserves typing/device/bot inputs and mobile keypad traces; only desktop challenge scoring changes. Typing in this cohort matches the CMU SVM schema. These results do not establish performance for an arbitrary-password account using the typing fallback.

Each public pointer center comes from that source identity’s separate three-minute TRAIN-support recording. DEV capture windows contain 641 consecutive original coordinates, never padded or joined across sessions. The endpoint uses the actual hash-pinned encoder and first five blocks. The simulator advances the challenge clock only in its disposable database; real-server clock checks remain active.

Across 100 additional deterministic artificial pairings, FAR ranges 10.00–13.33% (median 11.11%), and FRR ranges 25.00–41.67% (median 35.00%). This measures dependence on pairing, not a confidence interval or new participants. These secondary simulations assume usable captures and do not rerun the API.

V1 is retained: a mismatched adapter guard rejected stationary points in 19 recordings. V2 removes that invented guard while retaining the model, threshold, mapping and source coordinates. No biometric threshold tuning occurred; earlier exposure remains disclosed.

## Enrollment and use

Every desktop account without a neural pointer profile is guided to enrollment after a successful login. Three one-minute mouse/trackpad recordings are saved independently and form a per-account mouse profile. Existing typing and keypad profiles are preserved. The profile activates neural scoring only when desktop step-up is needed. Mobile uses the existing keypad. No user’s real pointer enrollment was fabricated.

Restart the pointer branch manually with `bash .worktrees/typing-pointer/experiments/run.sh` from the original repository root, then open `http://localhost:8006/pointer-enroll.html` after signing in. The script loads the checksum-pinned encoder from the original workspace. No server was started or restarted by this experiment.

## Reproduce

```bash
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/typing-pointer/experiments/neural_pointer/replay.py --version v2
BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/typing-pointer/experiments/neural_pointer/report.py
```

The replay refuses to overwrite a completed result. Source mappings, hashes, exact case outcomes and per-scenario rates are in `experiments/results/typing-pointer-neural-dev-v2.json`; all threshold and pairing results are in `typing-pointer-neural-tradeoff.json`.
