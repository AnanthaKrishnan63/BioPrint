---
title: Proposed strict HMOG BehaveFormer feature and training contract
tags: [hmog, behaveformer, protocol, proposal]
updated: 2026-09-20
---

# Proposed HMOG BehaveFormer contract

**Proposal only.** This audit read pinned author source and existing documentation; it did not read new participant archives, probes, scores or measurements, and did not train a model. A machine-readable contract must freeze unresolved choices below before any model fitting or selection. The intended claim is an unchanged author architecture with explicit preprocessing, sampling and small-budget adaptations—not a reproduction of published accuracy.

## Pinned source

[Author repository](https://github.com/DilshanSenarath/BehaveFormer), commit `319bba27196f18c089841491dc4bf57a1fe90453`:

- [HMOG preprocessing](https://github.com/DilshanSenarath/BehaveFormer/blob/319bba27196f18c089841491dc4bf57a1fe90453/data/HMOGDB/preprocess.py).
- [Accelerometer+gyroscope training and normalization](https://github.com/DilshanSenarath/BehaveFormer/blob/319bba27196f18c089841491dc4bf57a1fe90453/experiments/keystroke_imu_combined/HMOGDB/imu_acc_gyr/train.py).
- [Exact architecture](https://github.com/DilshanSenarath/BehaveFormer/blob/319bba27196f18c089841491dc4bf57a1fe90453/experiments/keystroke_imu_combined/HMOGDB/imu_acc_gyr/model.py).

Local source hashes and MIT license are retained under `references/hmog/`. Publisher schema provenance and its KeyPress column-order error are documented in [[hmog_schema_sources]].

## Exact ten keyboard channels

Let `p[i]` and `r[i]` be a complete key's press/release event times in milliseconds. Preserve this channel order:

| Index | Source name | Formula | Author scaling |
|---|---|---|---|
|0|hl|r[i] − p[i]|÷1000|
|1|di_ud|p[i+1] − r[i]|÷1000|
|2|di_dd|p[i+1] − p[i]|÷1000|
|3|di_uu|r[i+1] − r[i]|÷1000|
|4|di_du|r[i+1] − p[i]|÷1000|
|5|tri_ud|p[i+2] − r[i]|÷1000|
|6|tri_dd|p[i+2] − p[i]|÷1000|
|7|tri_uu|r[i+2] − r[i]|÷1000|
|8|tri_du|r[i+2] − p[i]|÷1000|
|9|key|Recorded KeyID|÷255|

The author code computes these over a reconstructed session before slicing windows. Missing final digraph/trigraph values are zero. It uses absolute `Systime` values as event times and can create zero-duration opposite events for incomplete pairs. Our proposal retains the formulas but uses verified native relative key-event times and excludes incomplete/ambiguous pairs. Negative UD from genuine overlap is not automatically invalid. Negative special-key IDs must not be silently replaced by guessed ASCII codes; keep the recorded numeric channel unless a separate mapping is explicitly frozen.

## Exact twenty-four IMU channels

The accelerometer occupies channels0–11, followed by gyroscope12–23. Each sensor uses this order:

`x, y, z, fft_x, fft_y, fft_z, fd_x, fd_y, fd_z, sd_x, sd_y, sd_z`.

The author preprocessing computes full complex FFT magnitude `abs(np.fft.fft(axis))`; first derivatives are `np.gradient(axis, edge_order=2)`, and second derivatives repeat that operation on the first derivative. No event-time spacing is passed to `gradient`: these are differences per sample index, not acceleration/angular-velocity derivatives per second. FFT coefficients retain the input length and are attached by row index; do not describe those rows as physically time-local spectra. Author source computes these features over the whole session before selecting sequence intervals.

Whole-session FFT on an allowed TRAIN session does not by itself access sealed sessions; it nevertheless uses context beyond an individual advertised window. Whole-session dev FFT also uses future observations from that dev session, so it is an offline protocol and cannot justify online-window latency claims. The proposed window-local adaptation avoids future-session context but waits for the complete30second window; centered derivatives and FFT remain batch features available only at window end. Larger FFT contexts also change spectral magnitude, preprocessing cost and memory, so timings are not interchangeable across these formulations.

Training-time scaling for the first24 channels is precise: accelerometer raw XYZ ÷10; accelerometer FFT XYZ ÷1000; gyroscope FFT XYZ ÷1000; other channels unchanged. The preprocessing file contains a `StandardScaler` helper, but the inspected pipeline does not call it. The author training script uses fixed divisions, not train-fitted z-scores. A new population scaler would therefore be an explicit adaptation, and must be fitted only on fit-cohort training windows—not selection, calibration, dev support or probes.

## Source windows, resampling and training

Keyboard input is50×10, with source stride5 and zero-padded final tail. The IMU sequence is100×24. Source IMU bin width is `(last_key_release − first_key_press)/100`, but bin iteration starts from the earliest selected sensor record. Its intervals include both endpoints, potentially counting boundary samples twice; empty bins become zero, excess bins truncate and short outputs pad. Preprocessing constructs all three sensors before the acc+gyro model keeps the first24 channels.

The architecture is unchanged: two five-layer spatio-temporal attention branches, twenty-component learned positional encoding, dropout0.1, 64D per-branch projections, and128→64 fusion. There are3,376,714 parameters. Loss is the author's unsquared Euclidean triplet hinge with margin1; Adam learning rate0.001. Source sampling chooses two distinct sessions of one uniformly selected identity for anchor/positive and a different identity for negative; sequences are sampled within those sessions. Its session index bounds come from the first identity, which must be corrected for unequal session availability. Source configuration uses64 triplets per batch and100 batches per epoch, with validation EER deciding saved checkpoints.

The source performs three forward calls—anchor, positive, negative—before backward/update. A combined24-example forward for eight triplets changes BatchNorm batch statistics even with the same loss and architecture. The existing synthetic cost probe used the combined call. Choose and record the execution convention; preserving three calls is the closer training adaptation.

The source loss computes `sqrt(sum((a-b)^2))` directly, applies `relu(d_positive-d_negative+1)` per triplet, then takes the batch mean. It has **no epsilon or distance clamp**; do not substitute PyTorch's epsilon-bearing built-in triplet implementation and claim identical loss. The synthetic probes compile only the original loss class AST, preserving its exact formula/reduction without executing the training module's data-loading path. Zero-distance gradient degeneracy is possible with this formula; fail on nonfinite loss/gradients instead of silently modifying the objective.

## Proposed safe paired-recording adaptation

1. Honor the immutable identity/session roles. Fit global weights/transforms only on fit-role records; select checkpoints on selection-role data; calibration identities set final thresholds only. Dev enrollment supports personal templates only. Never execute the author's downloader, recursive extraction or original split routine.
2. Within each participant/session/activity, pair real key down/up events by validated relative event time and key identity, preserving supported overlap. Require exact, unique event-time/action matches to `TouchEvent_im` for both endpoints. Reject ambiguous/missing matches; do not nearest-neighbor-match, infer missing releases or invent touches. A model window retains the matched original touch `Systime` anchors, not key-log `Systime`.
3. Use nonoverlapping half-open recording-time windows, prospectively30seconds, anchored at recorded Activity start. Both matched keyboard contacts and acc/gyro records must belong to that same activity and interval. Do not cross activity/session boundaries or clock resets. Require matched press and release anchors to fall inside the interval; do not reuse a contact in two windows.
4. Suggested strict keyboard contract: at least52 complete matched key pairs inside the window; compute the ten source channels for the first50 pairs using the remaining two as observed lookahead. This supplies actual digraph/trigraph context without accessing a later window or manufacturing padding. Report the52-pair observation cost and missing-window coverage. This higher count is a prospective proposal, not a result-driven relaxation of earlier protocols.
5. Suggested IMU contract: compute the source12-channel operators independently inside each recording-time window, then average into exactly100 half-open equal-width bins separately for acc and gyro. Require at least three raw records per sensor to define second-order gradients and at least one record from each sensor in every bin. Reject missing bins rather than inventing zero movement or interpolating across unobserved gaps. This explicitly changes whole-session FFT/derivatives and source bin anchoring/overlap. No magnetometer is needed for this model.
6. Apply source fixed divisions. Prefer no additional population transform initially; if a population normalization is selected prospectively, fit its parameters only on fit-role training samples and freeze before selection. Do not recompute statistics per dev session or use dev enrollment to fit the global encoder.
7. Sample anchor/positive from different eligible fit sessions of the same identity and negative from a different eligible fit identity, using each identity's actual eligible session list. Require at least two eligible fit sessions for a sampled genuine identity. Do not fall back to same-session positives if this fails; report infeasibility/coverage.

## Recording-time alignment: permitted interpretation

The publisher PDF names `Systime` as an absolute timestamp in key, touch and sensor tables. It supplies a common **recorded absolute-time convention**, not evidence of hardware-synchronized acquisition, a precise epoch definition, bounded delivery latency, or clock stability. Therefore matched-touch Systime plus sensor Systime can define *same-recording-interval* fusion, subject to TRAIN clock checks. It cannot establish exact same-instant physical movement around a key event.

Use key relative event timestamps only for hold/digraph/trigraph durations and exact touch event matching. Use the matched touch's absolute recording timestamp for interval membership; use each IMU record's Systime for the same membership/bins. Never equate sensor-relative nanoseconds divided by1e6 with key/touch uptime milliseconds: Android's clocks differ in deep-sleep treatment. Native sensor EventTime may validate ordering/sampling, but it is not an unverified cross-modality alignment conversion. See [MotionEvent timing](https://developer.android.com/reference/android/view/MotionEvent#getEventTime()), [SensorEvent timing](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp), and [SystemClock](https://developer.android.com/reference/android/os/SystemClock).

Logging and callback delays can shift or reorder Systime observations, smear bin means, exclude boundary events and weaken fusion. Keyboard contact spans can be ordered physically by relative time while their log times arrive out of order. Duplicate Systime values do not imply simultaneous physical events. Reject windows with backward matched anchors or unexplained clock discontinuities; retain audit counts. Do not estimate a latency correction by maximizing signal correlation or recognition performance. If recording-time association is accepted, label the result that way and preserve this limitation throughout reporting.

## Parameters still requiring a frozen decision

- Confirm30second Activity-anchored windows, the52-observed-pair rule, and exact down/up anchor availability over authorized TRAIN records. Insufficient coverage is a reportable outcome.
- Specify deterministic ambiguous-repeat/overlap pairing rules and whether CANCEL/multitouch events invalidate contacts; distinguish `_im` from duplicate ordinary/temp streams.
- Fix source raw key-code treatment, nonfinite rejection, sensor sampling-gap ceiling if used, tie ordering for equal Systime, and the policy for clock discontinuities. Do not derive thresholds from dev.
- Confirm window-local FFT/gradient operators before bin averaging versus an explicitly different resample-first formulation; these are not mathematically interchangeable. Preserve sample-index derivative units.
- Decide no population normalization versus fit-only normalization before selection. Freeze any zero-variance handling and prohibit per-probe fitting.
- Freeze checkpoint-selection metric, verification gallery size, distance rule, aggregation and calibration operating points. Author full-session evaluation cannot simply be relabeled as the proposed30second protocol.

## Prospective bounded CPU schedule

A minimal architecture experiment could use one deterministic seed,20epochs,40batches/epoch and8triplets/batch on two CPU threads, with unchanged architecture, source loss and Adam0.001. Use the author's three separate8-example forward calls so BatchNorm follows the source convention. This is800 updates/6,400 sampled triplets, deliberately smaller than the author schedule. No hyperparameter search or dev-based extension is proposed. Use selection-only checkpoint choice and freeze before calibration/dev. The existing synthetic combined-forward benchmark suggests about5.49minutes optimizer computation; three-call execution, data loading, validation and real sparsity can change that cost. This is feasibility evidence only, not an accuracy forecast or authorization to start training.

The separately authorized three-forward synthetic probe now measures median0.58909seconds per update,95th percentile0.61663seconds, and peakRSS560,893,952bytes (about535MiB). All13 warmup/measured steps had finite loss, gradients and updated parameters. The same illustrative800 updates extrapolate to7.85minutes of computation. This supersedes the combined-forward number for the proposed execution convention; both reports and their hash-pinned scripts are preserved. New files: `scripts/behaveformer_cpu_training_probe_separate.py` and `references/hmog/behaveformer_cpu_training_probe_separate.json`. No real data, pretrained checkpoint or trained weight artifact was accessed/persisted.

Related: [[hmog_schema_sources]], [[hmog_protocol_proposal]].
