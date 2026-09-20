---
title: HMOG schema evidence and reproducible method audit
tags: [hmog, schema, clock, multimodal, reproducibility]
updated: 2026-09-20
---

# HMOG schema evidence and method audit

Scope: read existing protocol/source notes, primary publications and author code. With separate authorization, retrieved **only publisher documentation**, not participant ZIP contents. Subsequently read only root's already guarded first-fit TRAIN probe, `datasets/hmog/train_schema_probe.json`; no other observations, dev/test records, or subject ZIP members were read here. No model was trained.

## Author documentation and provenance

The [official HMOG repository](https://github.com/hmog-dataset/hmog) links the public publisher archive. Its separately indexed `public_dataset/data_description.pdf` was retrieved with exact HTTP206 ranges using `scripts/hmog_acquire.py`'s response verifier. Compressed payload range6132250002–6132348587:98,586 bytes; decompressed PDF105,821 bytes; CRC32 `444708416`; SHA256 `0ddf1993f4c06f0215ec34f127813311856e3f88460b0996c009c304aac1a57e`. No other archive member was fetched.

Local files: `source_documents/hmog_data_description.pdf`, its `.receipt.json`, extracted `.txt`, and page5 visual snapshot `hmog_keypress_schema.png`. Table order was visually checked. Treat this PDF as author documentation with a demonstrated ordering error below, not unquestionable ground truth.

## Exact fields and evidence boundaries

All indices below are zero-based. The PDF supplies names/descriptions; actual KeyPress field order is supported by the authorized TRAIN probe and an independent paper-author parser.

| Stream | Field order | Evidence/status |
|---|---|---|
| `KeyPressEvent.csv` |0 Systime;1 PressTime;2 ActivityID;3 PressType;4 KeyID;5 Phone_orientation|Actual TRAIN row and BehaveFormer parser agree. **PDF lists PressType before ActivityID**, contradicting both; do not implement that PDF order.|
| `TouchEvent.csv` |0 Systime;1 EventTime;2 ActivityID;3 Pointer_count;4 PointerID;5 ActionID;6 X;7 Y;8 Pressure;9 Contact_size;10 Phone_orientation|PDF section5; first TRAIN rows match the documented structure.|
| `TouchEvent_im.csv` |Same11-field structure in inspected TRAIN rows|Not described by the PDF. Its first down event exactly shares the key-down relative timestamp, supporting keyboard-touch association in that TRAIN fragment only. Global completeness, duplication and coordinate frame remain unverified.|
| `tempTouchEvent.csv` |Same11-field structure in inspected TRAIN rows|Not described by the PDF. All five inspected rows duplicate `TouchEvent.csv`; do not concatenate these streams and count duplicates as independent contacts.|
| `Activity.csv` |0 ID;1 SubjectID;2 Session_number;3 Start_time;4 End_time;5 Relative_Start_time;6 Relative_End_time;7 Gesture_scenario;8 TaskID;9 ContentID|PDF sections1; TRAIN probe matches. Activity ID is a composite of subject/session/content/runtime counter, not a continuous numeric feature.|
| Accelerometer/Gyroscope/Magnetometer |0 Systime;1 EventTime;2 ActivityID;3 X;4 Y;5 Z;6 Phone_orientation|PDF sections2–4. Native sensor timestamp units differ from touch/key in inspected TRAIN.|

Key `PressType`: **0 down,1 up**. Touch `ActionID`: **0 or5 down;1 or6 up;2 move**. Pointer_count distinguishes single/multiple contacts; PointerID0 is first/single pointer and1 second pointer in the documented case. Do not merge simultaneous pointer tracks. Orientation:0 portrait,1 rotated90°counter-clockwise,3 rotated90°clockwise. Pressure/contact-size scale and coordinate origin/units are not completely specified by this PDF; do not invent physical pressure units or assume `_im` and ordinary touch share a coordinate frame.

Activity `Gesture_scenario`:1 sitting,2 walking. `TaskID` groups are1/7/13/19 reading+sitting;2/8/14/20 reading+walking;3/9/15/21 writing+sitting;4/10/16/22 writing+walking;5/11/17/23 map+sitting;6/12/18/24 map+walking. ContentID1/2/3 selects the subtask. **TaskID is not session ordinal**: the permitted probe has session3 and TaskID22. Use recorded TaskID, not filename arithmetic, to identify writing/motion context.

## Clock semantics: what is established

The PDF describes Systime as absolute, key PressTime/touch EventTime as relative, and Activity relative times as since boot. It does **not** explicitly specify every stream's unit, absolute epoch, logger call, or deep-sleep behavior.

The guarded TRAIN fragment establishes a shared key/touch `_im` relative timestamp at a down event. The corresponding key up is125 relative units later. Rows are logged in an order that reverses those physical events: the up precedes its down in file/Systime order. Therefore, row order or Systime differences cannot substitute for physical key hold timing. Pair by validated event time, key and activity; never fabricate missing releases. BehaveFormer's choice to rename its internal down/up labels is merely an encoding convention, not itself an error.

Android's primary APIs distinguish [MotionEvent.getEventTime](https://developer.android.com/reference/android/view/MotionEvent#getEventTime()), in uptime milliseconds, from [SensorEvent.timestamp](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp), in elapsed-realtime nanoseconds. [SystemClock](https://developer.android.com/reference/android/os/SystemClock) documents that uptime excludes deep sleep while elapsed realtime includes it. Thus dividing sensor time by1e6 resolves units but **does not prove a shared clock origin**. These API facts are not proof that HMOG's unpublished logger used every API unchanged.

The TRAIN sensor magnitudes support a nanosecond interpretation; Activity/key/touch relative values support milliseconds. Root must validate this over allowed TRAIN sessions, including resets, wall/event offsets, drift and explicit key/touch anchors. A constant sensor–touch offset cannot be inferred from row alignment or optimized on dev. Until clock provenance/synchronization is established, typing+touch may proceed under its verified shared clock while IMU fusion remains gated. Do not normalize each modality independently to its first row.

## Original HMOG baseline:2016, not current SOTA

The [original paper](https://arxiv.org/html/1501.01199#S4) defines96 inertial features:60 grasp-resistance and36 stability measurements around taps. Its tap comparator uses11 features:duration, nine contact-size summaries, and velocity between consecutive tap starts. Keystroke comparators use89 key-hold and1,225 common-key digraph features. Scaled Manhattan/Euclidean templates use enrollment feature means and user standard deviations; one-class RBF SVM is another comparator. Training cross-validation selects preprocessing and fusion weights. Test authentication vectors average observations across20–140second scans. Portrait-only writing sessions and separate sitting/walking evaluations are used; first two sessions train, remaining two test per condition. This differs from our fixed identity/session-role proposal.

**CPU-feasible starting point:** faithfully implement the documented tap features, key timing groups, enrollment mean/std-scaled Manhattan distance and weighted score fusion once event schema is verified. Freeze population preprocessing/fusion on our assigned TRAIN roles. This preserves a source method family; changed cohort, support budget,30second windows, missing-feature policy and split make it a protocol adaptation, not reproduced paper performance. A generic pressure/velocity summary vector must not be labeled the original HMOG96-feature extractor.

## Modern attributable code: BehaveFormer

[BehaveFormer, IJCB2023](https://arxiv.org/abs/2307.11000), [author code](https://github.com/DilshanSenarath/BehaveFormer), combines **typing and IMU**, not a jointly trained typing+touch branch. Source snapshot commit`319bba27196f18c089841491dc4bf57a1fe90453` and MIT license are retained under `references/hmog/`. The [2024 author extension](https://github.com/nganntk/BehaveFormer) also evaluates swipe+IMU; separate typing and swipe experiments do not establish simultaneous typing+touch fusion. Neither is demonstrated here as current2026SOTA.

The inspected2023 accelerometer+gyroscope model has:

- 50×10 key input and100×24 IMU input; five spatio-temporal encoder layers per branch; learnable20-component positional encoding.
- Temporal and channel multihead attention; key branch5/10 heads, IMU6/10 heads; residual LayerNorm and1×1,3×3,5×5 convolution blocks; dropout0.1.
- Key MLP500→250→64; IMU2400→1200→64; concatenation128→64. Exact author class has **3,376,714 parameters**.
- Euclidean triplet margin1, Adam0.001; configured64 triplets/batch and100 batches/epoch. Published repository dependencies pin Python3.10.12/PyTorch2.0.1+CUDA11.8 plus NumPy/Pandas/SciPy/sklearn. The architecture alone successfully imports and runs on existing CPU PyTorch2.14.

Generated-zero forward checks only, CPU2: batch1 median17.89ms; batch8 median40.54ms; finite64D outputs. `references/hmog/behaveformer_cpu_synthetic.json` records this audit. No training-time estimate or recognition accuracy follows from inference timing. A faithful architecture run is computationally plausible; full training feasibility needs a separately authorized synthetic backward/epoch benchmark and a frozen data contract.

A subsequent authorized synthetic training-cost probe now measures the exact retained architecture and exact author unsquared Euclidean triplet loss (margin1), using Adam0.001 and8 triplets/24 examples per step. Three warmup and ten measured steps onCPU2 give median0.41156seconds,95th percentile0.43140seconds, peakRSS527,216,640bytes; all losses, gradients and updated parameters remain finite. An illustrative20epochs×40batches would require roughly5.49minutes of optimizer computation at this batch size, excluding preprocessing, validation and I/O. This is not the author's64-triplet full training budget or an accuracy prediction. No real measurements/checkpoints were read and no updated weights persisted. `scripts/behaveformer_cpu_training_probe.py` and `references/hmog/behaveformer_cpu_training_probe.json` preserve the runnable procedure, deterministic seed and source hashes. Dataset/clock/preprocessing gates remain unresolved by this hardware test.

**Do not run author preprocessing unchanged.** It recursively extracts all ZIPs, reads all sessions, inserts synthetic opposite key events when pairs are missing, uses Systime rather than relative event times, selects overlapping50-key windows with stride5, and pads tails. Those operations conflict with this project's sealing and no-fabrication rules. Correcting them creates a documented preprocessing adaptation. Official README links pretrained weights, but none were fetched: HMOG-trained weights could have already seen our held-out identities, so source subject provenance must be verified before any transfer evaluation.

## Remaining gates

Resolve `_im`/ordinary/temp stream identity and duplication over TRAIN, validate key/touch event matching and coordinate frames, and document the PDF column error in parser assertions. Keep inactive/insufficient accounts in coverage reports. Confirm sensor-clock relation before invoking IMU features. Preserve source licenses, HMOG nonredistribution restrictions, and the required William-and-Mary disclaimer. The College of William and Mary is not responsible for this analysis or interpretation.

Related: [[hmog_protocol_proposal]], [[bbmas_hmog_feasibility]], [[keystroke_sources]].
