# Follow-up: transferable matcher over aligned personal residuals

Exploratory follow-up declared after the summary experiment's fixed-text DEV
results showed 96–99% FRR. Those results remain unchanged. This reused DEV is
validation with prior exposure, not a pristine test or independent confirmation.

A password is fixed within one account even though it differs across accounts.
Fit the existing positional profile from ten enrollment recordings. Align each
attempt to its owner's template, standardize deviations using that profile's
spread, and summarize deviations separately for hold/DD/UD families. A shared
matcher therefore has fixed input size without requiring fixed password text.
For each family: five absolute-residual quantiles, mean, standard deviation,
signed median, signed mean and signed standard deviation, clipping residuals to
[-6,6]. Append the existing distance divided by the enrollment-derived threshold.
This produces 31 features regardless of password length or key names.

Use exactly the earlier CMU/KeyRecs-fixed eligible recordings and identity roles.
Free text is ineligible because it does not repeat a personal password template;
retain its separate summary-model results. Do not align unrelated free texts.
First ten TRAIN recordings are enrollment; remaining fit-identity TRAIN attempts
fit shared models, calibration identities set thresholds at empirical FAR<=1%,
and evaluation identities' DEV attempts test them. No test reads.

Fit balanced logistic regression C=1 and 128-tree ExtraTrees with leaf size10,
seed20260920, two CPU threads. Balance tracks and classes. No parameter search.
Use pooled, CMU-only and KeyRecs-only training. Freeze all six learned models
and source-only thresholds before this follow-up evaluates DEV. Add the original
distance/threshold ratio recalibrated on the same source calibration as a fair
operating-point baseline. Report all nine variants, exact counts and EER.
No automatic deployment or online adaptation. Cross-source transfer covers only
two fixed password texts; arbitrary-password accuracy remains unestablished.
