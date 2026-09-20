# HMOG tap11: prospective same-window reference

This proposal is frozen before new tap measurement access; it authorizes no reads. Machine-readable choices and input hashes are in `hmog_tap_protocol_proposal_v1.json`. Existing evidence is TRAIN-only: fit-v3 contains 207 typing/IMU windows; this establishes the denominator, not tap feasibility. The earlier strict consecutive-key protocol failed and was retained. That failure does not authorize gap-filtered tap velocity. No DEV outcomes informed these choices.

## Published family and explicit adaptations

The [original paper, IV-A/B/C](https://arxiv.org/html/1501.01199#S4) specifies eleven tap descriptors, enrollment means/user standard deviations, and averaged scan vectors. Its tap comparison selects three features through training mRMR: duration, average contact size, and inter-press velocity. It uses an 80-observation enrollment minimum. We retain all eleven as the primary source-family reference. An optional fixed published-three ablation is source-informed, not a new mRMR execution: the paper does not fully establish implementation/discretization details needed for exact feature-selection reproduction.

Required arithmetic: event duration in milliseconds, Euclidean down-to-down displacement divided by elapsed seconds in pixels/second, raw contact-size mean/median/std/Q1/Q2/Q3/first/min/max. Preserve median/Q2 redundancy. Use linear sample quantiles and population std (`ddof=0`), equal sample weights including endpoints. These are documented numerical conventions, not exact reproduced implementation or current SOTA.

## Exact windows and raw contact eligibility

Copy **only metadata** from the frozen fit-v3 archive: `(subject, session, activity_id, window_index)`. Reconstruct its existing Activity-anchored 30-second half-open recording intervals; never reslice, shift, or create replacement windows. Preserve every original row even when taps are unavailable. Use original `TouchEvent_im.csv` only, retaining complete DOWN/MOVE/UP samples; do not concatenate ordinary/temp touch streams.

Build an unfiltered event ledger alongside `parse_contacts`. A completed-contact list alone cannot prove adjacency because the parser discards canceled, missing, and invalid cycles. Each eligible contact must be unique, uncancelled, finite, portrait, and have pointer count exactly one throughout. Event milliseconds strictly increase; size is nonnegative. Both previous and current contacts must belong to the same activity, coordinate stream/frame and session, be nonoverlapping, and lie wholly inside the same recording window. Never borrow a predecessor from an earlier window.

Choose the immediately preceding **observed** contact, never the previous surviving valid contact. Cancellation, duplicate/repeated/unknown/orphan events, missing endpoints or intervening invalid contact break the chain. Ambiguous equal-time event order is rejected. First contact and broken chains yield no velocity/vector. No zero imputation, timing correction, contact stitching or physical-area conversion. Completely unrecorded contacts cannot be detected; observed adjacency is not proof against every logging omission. This proposal uses valid keyboard-stream touches within existing paired windows; it does not add mandatory one-to-one key-event matching to the published tap comparator.

## Aggregation, scoring and coverage

Require at least **five valid tap11 vectors** per existing window, prospectively fixed. The window vector is their arithmetic mean. Enrollment templates use individual valid tap vectors from the assigned gallery windows, not means of those windows. Require **80 valid enrollment tap vectors per account**. Ignore dimensions with zero enrollment spread; report the mask/count and abstain if none remain. Score with summed absolute mean/std-scaled differences; larger means less genuine. Never fit templates on probes or population transforms on DEV support.

Retain per-window counts: observed downs, completed contacts, vectors, reasons for missing predecessor, cancellation/overlap/duplicate/other rejection, and original-window eligibility. Store flat tap vectors plus offsets keyed to original rows and an explicit scorable mask; do not align by array position after filtering. Report original denominator, tap-intersection denominator, enrollment failures and all account coverage.

## Loader and comparison gates

A new fit-only `tap_reference_feature_extraction` purpose/phase should pin this finalized protocol and exact window archive. Permit only the four fit identities, preserving enrollment sessions1–8 versus train_fit9–16. Selection/calibration/DEV require separate authorization and frozen recipes; test sessions17+ remain sealed. Existing generic schema/extraction permissions must not silently authorize this new experiment.

Root should freeze classical comparison and fusion before selection: tap11 alone, existing typing/IMU comparator on identical window IDs, and an explicitly defined three-modality combination. Use aligned same-person/same-window signals for genuine and impostor claims. Publish both shared-intersection metrics and end-to-end coverage; a smaller easy subset must not appear as unconditional improvement. This is a follow-up on a DEV cohort already evaluated for the encoder, so any later validation must disclose prior exposure and cannot claim a pristine new test.
