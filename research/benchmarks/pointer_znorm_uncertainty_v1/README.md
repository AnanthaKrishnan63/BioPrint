# Conditional uncertainty for the frozen pointer comparison

This post-hoc analysis uses only the existing frozen SapiMouse DEV score artifact.
The plan and source were saved before bootstrap execution. No models, thresholds,
normalization statistics, or selection rules were fitted or changed.

| Metric | Cosine point estimate | Z-norm point estimate | Paired difference 95% interval, Z-norm minus cosine |
| --- | ---: | ---: | ---: |
| FAR | 1.66% | 0.74% | −1.63 to −0.24 percentage points |
| FRR | 43.42% | 60.53% | −10.23 to +44.29 percentage points |
| Descriptive pooled EER | 13.87% | 11.73% | −5.47 to +3.98 percentage points |

The apparent EER reduction is uncertain. The conditional FAR interval is below
zero, while the FRR difference remains imprecise. Cosine remains selected by the
original TRAIN-only rule; these intervals cannot justify DEV-based promotion.

## Method and limitations

Two thousand paired bootstrap replicates resample 24 probe-person clusters with
replacement, preserving each person's decisions against all enrolled accounts.
Both scorers use identical draws; rates are claim-weighted as in the original
report. All 76 decisions and 1,824 claims remain in the observed comparison.
FAR/FRR use each method's original threshold targeting 1% TRAIN FAR. EER follows
the original nearest-crossing ROC definition and never supplies a new threshold.

These percentile intervals condition on fixed enrollment references and models.
They do not resample training or shared reference-account uncertainty, and do not
repair previous DEV exposure. They are not population-wide or pristine-holdout
confidence claims. No new raw data or test observations were inspected.

`bash scripts/research.sh scripts/pointer_znorm_uncertainty.py prepare` freezes
source and input hashes; `run` verifies them and refuses to overwrite results.
Both commands have already completed. `results.json` contains unrounded rates
and intervals; `source.py` preserves executed code. Two generated-data tests pass,
including equivalence between weighted claims and explicit replication. The
unchanged scoring APIs retain their earlier exact replay evidence; this analysis
does not constitute new all-feature validation.
