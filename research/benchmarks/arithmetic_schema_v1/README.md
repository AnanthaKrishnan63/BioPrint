# TRAIN-only arithmetic schema audit

All 105 permitted files from seven TRAIN-fit people were decoded under the
frozen acquisition/schema plans. Calibration, DEV, test, practice, and other
task members were excluded by an explicit whitelist. Archive/source/input
hashes were checked before access.

Each file has five presented trials. Twenty-eight files also have an empty
sixth timestamp slot with zero correctness; this is padding, not a missed trial.
Across 525 presented trials there are 470 measured responses, 55 missing
responses, and 373 correct answers. All measured response times are positive,
between 0.5865 and 2.4871 seconds. Missing counts by low/middle/high level are
1/11/43 out of 175 trials each. These are TRAIN diagnostics, not validation rates.

`../arithmetic_diagnostics_v1/` retains per-slot diagnostics and executed source.
`arithmetic_features.py` preserves missing response times as `None`. Bounded
waiting time uses the stated 2.5-second deadline for a miss, explicitly not an
observed RT. Separate missing and incorrect fractions retain that distinction.
Invalid timestamps, ambiguous padding, non-monotonic stimuli, and inconsistent
correctness are rejected rather than silently removed. Every original TRAIN
file passes this parser; `../arithmetic_parser_validation_v1/` records that check.

Candidate representations are mean bounded wait, ordinal slope, three condition
means, and a nine-component wait/missing/error profile. They are definitions,
not selected or optimized models. With levels coded 0/1/2, the OLS slope equals
half the high-minus-low contrast and discards the middle-level mean. Fixed task
order, deadline censoring, one visit, and aggregation across separate blocks
limit interpretation. A model/threshold selection protocol must precede
calibration and DEV access.

Twelve generated acquisition, scope, and parser tests pass. An all-timeout
fixture initially exposed loss of empty cells during array flattening; the
parser now preserves each empty slot. No dataset criterion was changed using
DEV, and no FAR/FRR/EER or SOTA claim follows from this schema audit.
