# typing-pointer

Shared baseline: `0e11268`. This branch is experimental; no population FAR guarantee.

Policy version 2 uses the full enrolled cognitive threshold on mobile (`mobile_keypad_ratio=1.0`), instead of the earlier 0.75 multiplier. Desktop target checks retain 0.75. Bot checks, missing-evidence rejection, and the hard typing limit still apply. This change responds to live usability failures; it is not a new calibrated FAR result. Historical synthetic reports predate this change and the removal of the invalid fast-keypad-search bot rule.

This branch uses strict typing admission and adaptive verification. Desktop accounts with a neural pointer profile now use the trained SapiMouse FCN encoder and cosine matching during pointer step-up. Three one-minute recordings at `/pointer-enroll.html` create each account's profile. Without this profile, the previous run-level motor scorer remains the fallback; mobile uses cognitive keypad timing. Direct typing admission is unchanged. The original four-combination benchmark predates this neural integration; see `experiments/NEURAL_POINTER_RESULTS.md` for the follow-up.

The pointer launcher sets `BIOPRINT_POINTER_ENCODER`. Its SHA-256 is pinned, and the cosine threshold is 0.9480821490287781, from source TRAIN calibration at a 1% FAR target. This target is not an app-wide FAR guarantee. Enrollment adds tables without replacing existing profiles or databases. Verification does not update profiles.

Supervised typing requires a checksum-pinned TRAIN background bank whose feature names exactly match the password. Otherwise the existing enrollment-distance scorer is used and identified in the signal explanation. C=10 and gamma factor=0.1 are fixed; calibration is separate from fitting. No probe updates any profile.

Use `BIOPRINT_DB` pointing to an isolated branch database. Do not point this branch at production `bioprint.db`. Use the bigidea environment. Background data is set with `BIOPRINT_BACKGROUND` and `BIOPRINT_BACKGROUND_SHA256`. Threshold ratios are in `experiment.json`. API metadata: `/api/experiment`.

No listener was started by setup. Synthetic evaluations use the same frozen input files for every branch; challenge IDs differ but layouts, targets, timing and decision inputs must agree.
