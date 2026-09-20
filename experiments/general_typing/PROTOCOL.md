# Password-independent matcher experiment

Declared before fitting or inspecting new results, 2026-09-20.

Use existing guarded CMU and KeyRecs fixed/free TRAIN and DEV loaders. Never
decode sealed identities or CMU sessions 7–8. Previously explored DEV is not a
pristine test set. No new downloads, production database access, or server.

The shared model compares an enrollment profile with an attempt, rather than
classifying a fixed list of people or requiring a particular password. Every
record becomes 21 distribution features: seven statistics (10/25/50/75/90th
percentiles, mean, standard deviation) of hold, down-down and up-down times in
milliseconds. No key names, text, participant identifiers, dataset labels or
password length enter the learned model. Free text uses nonoverlapping nine
digraph windows, a short-input stress test rather than a password trial.

Use the first ten TRAIN records for personal enrollment. Remaining TRAIN records
of fit identities fit the matcher; those of distinct calibration identities set
thresholds. Evaluate only DEV recordings of a third, disjoint identity group.
Assign identities deterministically by SHA256 of dataset namespace plus ID;
50% fit, 25% calibration, 25% evaluation. KeyRecs fixed/free share identity roles.
All eligible records are used; require at least 11 TRAIN records and ten finite
enrollment vectors. Exclude nonfinite/negative hold observations and count them.

Candidates fixed in advance: robust profile distance; balanced logistic
regression (C=1); ExtraTrees (128 trees, minimum leaf 10, balanced labels,
seed 20260920). Inputs to learned matchers are absolute standardized deviations
from the enrollment median (spread floor 10 ms and 10% of median), plus signed
deviations. Fit scaling only on matcher training data. No hyperparameter search.
Weight each source track and genuine/impostor class equally when fitting.

Run pooled training, CMU-only transfer to KeyRecs, and KeyRecs-only transfer to
CMU. Freeze each model and its empirical calibration-FAR<=1% threshold before
DEV access. Transfer thresholds use only source datasets. All-pairs comparisons
stay within a dataset/track; no identities are joined across datasets.

Report pooled FAR/FRR with integer numerators/denominators, diagnostic empirical
EER, per-track results, actual profile/probe counts, exclusions, model hashes and
runtime. Compare candidates on identical trials. No threshold retuning or
deployment selection on DEV. Short free text is not evidence that an impostor
knowing a password can be rejected. Desktop results do not establish mobile
performance. Personal enrollment remains required for every new account;
automatic online adaptation is outside this experiment.
