# typing-pointer

Shared baseline: `0e11268`. This branch is experimental; no population FAR guarantee.

The control retains its original decision policy. Candidates use strict typing admission, three extra typings or two scrambled target runs. `typing-pointer` uses the existing run-level mouse motor profile on desktop and cognitive timing on mobile; `typing-keypad` uses cognitive timing. These are existing feature-based keypad models, not the SapiMouse encoder.

Supervised typing requires a checksum-pinned TRAIN background bank whose feature names exactly match the password. Otherwise the existing enrollment-distance scorer is used and identified in the signal explanation. C=10 and gamma factor=0.1 are fixed; calibration is separate from fitting. No probe updates any profile.

Use `BIOPRINT_DB` pointing to an isolated branch database. Do not point this branch at production `bioprint.db`. Use the bigidea environment. Background data is set with `BIOPRINT_BACKGROUND` and `BIOPRINT_BACKGROUND_SHA256`. Threshold ratios are in `experiment.json`. API metadata: `/api/experiment`.

No listener was started by setup. Synthetic evaluations use the same frozen input files for every branch; challenge IDs differ but layouts, targets, timing and decision inputs must agree.
