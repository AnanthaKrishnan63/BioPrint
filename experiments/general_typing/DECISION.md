# General typing model: integration decision

We built and trained password-independent matchers, but **do not replace the login
scorer with them yet**. Their measured genuine rejection rates are too high.

The four earlier branches share identical typing extraction code. For a password
other than the exact CMU feature schema, their underlying typing score falls back
to the existing personal distance model. Policy and challenge routing still differ.

## What was built

- A summary-based shared matcher using 21 timing-distribution features and a
  personal enrollment profile. Trained on CMU and KeyRecs fixed/free recordings;
  also evaluated frozen transfer to BEACON gameplay keyboard recordings.
- An improved architecture that first aligns each attempt to **that account's own
  password**, then learns from 31 residual features. Different users can have
  different password texts and lengths; no common `.tie5Roanl` requirement.
- Twelve trained classifiers (logistic and tree models across pooled and
  single-source runs), six calibrated distance controls, and an offline adapter
  accepting browser-format enrollment and attempt recordings.

## Results that matter

The aligned pooled logistic model is a compact example, not a selected deployment
winner. These are held-user, single-attempt typing results:

| Dataset | Existing scorer EER | Aligned logistic EER | New FAR | New FRR |
|---|---:|---:|---:|---:|
| CMU | 26.31% | 25.41% | 0.42% (38/9,000) | 95.30% (953/1,000) |
| KeyRecs fixed | 31.64% | 29.42% | 3.83% (695/18,161) | 88.98% (1,243/1,397) |

The aligned pooled tree model reaches EER 24.19% on CMU and 30.78% on KeyRecs;
its FAR/FRR are 1.72%/90.90% and 4.58%/82.61%, respectively. This also does not
justify promotion. The simpler distribution-only model performed worse overall;
its fixed-text FRR reached 96–99% at the strict threshold.

Thresholds were calibrated to empirical 1% FAR using separate identities, then
frozen before evaluation. That target did not consistently transfer to unseen
users/datasets. Lower FAR alongside enormous FRR is not successful authentication.
The original scorer's operating threshold differs; the full aligned report also
includes a recalibrated original-distance comparator to separate threshold effects.

## Evidence and limits

- [Summary experiment](RESULTS.md): all 36 evaluations, exact counts, exclusions,
  gameplay transfer, runtime and model hashes.
- [Aligned experiment](ALIGNED_RESULTS.md): all 18 follow-up evaluations, formulas,
  baseline comparisons, transfer results and reproduction commands.
- Fifteen tests passed; independent audits recomputed 54 score tables and checked
  18 artifact hashes and identity separation. Artifacts remain available locally;
  large model binaries and score arrays are excluded from Git.

No live database, enrollment, server or original experiment branch was modified.
This is not an all-feature app evaluation and not an online-adaptation experiment.
The follow-up reused previously inspected DEV, explicitly disclosed. Source data
cover only two fixed password texts; supporting arbitrary input is not evidence of
accuracy for every password. More diverse repeated-password evaluation is needed
before a general matcher can be trusted for direct admission. Existing step-up
signals remain necessary while that work is unresolved.
