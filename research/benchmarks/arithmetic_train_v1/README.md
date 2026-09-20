# Frozen arithmetic TRAIN comparison

Eight candidates combine four representations with scaled Manhattan or
shrinkage Mahalanobis distance. The latter uses
[Ledoit–Wolf covariance estimation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)
on within-account enrollment residuals. This is a source-backed statistical
adaptation, not an established SOTA arithmetic-authentication method.

Seven TRAIN-fit people supply T2/T3 enrollment for global scales/covariance.
T4 determines an internal 1% FAR-target threshold; T5/T6 select the method,
preferring observed FAR≤1%, then FRR/EER/fixed order. Selection is written
before opening the four separate TRAIN-calibration people. Their T2/T3 form
personal templates; T4–T6 determine final thresholds without method reselection.
Each template uses 30 trials; a probe combines 15 trials across three separate
difficulty blocks. This is not a three-question login or a cross-day study.

| Representation / distance | Selection FAR | Selection FRR | Selection EER |
| --- | ---: | ---: | ---: |
| Mean wait / either distance | 9.52% | 85.71% | 44.05% |
| Ordinal slope / either distance | 1.19% | 92.86% | 36.90% |
| Condition means / Manhattan | 0% | 92.86% | 39.29% |
| Condition means / Mahalanobis | 0% | 100% | 32.74% |
| Full wait/missing/error profile / Manhattan | 0% | 78.57% | 24.40% |
| Full profile / Mahalanobis | 0% | 100% | 35.12% |

The full-profile Manhattan method remains selected. At its final calibration
threshold targeting 1% FAR, actual TRAIN-calibration FAR is 0/36 and FRR is
11/12 (91.67%); descriptive EER is 31.94%. Zero observed calibration FAR is
partly by construction and cannot establish a low population FAR. Condition
means have lower calibration FRR, but the protocol forbids reselection there.
No near-zero-error or universal improvement claim is supported.

`arithmetic_train.py prepare/run` preserves plan, source, selection, scores,
parameters and final thresholds. Generated tests verify role rejection, score
axes against hand calculations, tie-safe low-FAR calibration and failure-aware
selection. Related feature/scope tests also pass.

## DEV outcome

The frozen eight-method DEV run failed its required enrollment schema before
producing scores. A diagnostic using the unchanged parser checked all60 blocks:
`Cal_LWS_LlT2.mat` contains 50 entries in all three arrays, versus five expected;
59 other blocks pass. No truncation, replacement, participant removal, threshold
change, or subset scoring occurred. DEV FAR/FRR/EER are **unavailable**. See
`../arithmetic_dev_failure_v1/report.json`. This is an evaluated enrollment
failure, not successful complete-cohort recognition validation. API score replay
and all-feature fusion remain incomplete; sealed-test archives remain absent.
