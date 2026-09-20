---
title: Type2Branch reproduction feasibility audit
tags: [keystroke, reproducibility, feasibility]
updated: 2026-09-20
---

# Type2Branch feasibility decision

**The pinned author encoder is now implemented and evaluated through a documented adaptation; this is not an exact published SOTA reproduction.** The original feasibility audit below is historical. Subsequent work resolved local execution and performed TRAIN-only fitting/selection/calibration followed by a disclosed cross-session DEV protocol. Source synthesis, residual, sequence-order and capture-budget differences remain explicit.

## Executed short-capture follow-up

The paired TRAIN study selected mixed-prefix epoch4/500 updates. Both arms used
the same initial checkpoint and identical unaugmented batch stream. Reserved
TRAIN calibration set exact-length thresholds before DEV scoring.

All79active accounts were eligible:78used100events and one used25events.
DEV FAR was **563/6162 = 9.13664%**, FRR **24/79 = 30.37975%**, and pooled
discrete EER **13.93216%**. The empirical1% calibration FAR target did not
transfer across sessions. The actual API worker replay reproduced all6241
scores and every reported metric exactly, with zero decision flips.

See `type2branch_capture_dev_evaluation_v1`, `type2branch_capture_api_replay_v1`
and `type2branch_length_pair_v1`. This follow-up preserves the original failed
1000-event protocol. Its prior DEV exposure, one-probe-per-identity precision,
TRAIN-role exposure and different capture budget prevent pristine-test or
direct-comparison claims. It does not validate all-feature fusion.

## Follow-up: external generator located

The [published paper](https://repositorio.uam.es/server/api/core/bitstreams/17c2ba94-8f23-40af-a48b-68a01a6b4fbf/content)
explicitly links the external implementation through its software reference.
The [Software Impacts archive](https://github.com/SoftwareImpacts/SIMPAC-2022-276)
is a fork of `lsia/ksdsld`; it provides C# finite-context synthesis source under
GPLv3. Metadata, README and license are pinned locally at tree
`1fdc0e641a4ca6f817c8417a2c69bb1f1b747b14` in
`references/type2branch_synthesis/`. The earlier observation that the model
repository lacks the producer is correct, but does **not** establish that the
producer is unavailable. This changes the next action to inspecting and
validating that external implementation.

Paper equations7–9 define the two extra channels as observed hold/flight timing
minus population-synthesized timing, alongside the key code and ordinary timing
channels. The population profile must use only our TRAIN partition; the paper's
word "development" does not authorize using our DEV probes. The local merger
only concatenates the supplied CSV columns, so the caller must establish their
residual transformation, units, normalization and clipping. Simply appending
raw synthetic timings would not prove paper-equivalent features.

The external README names several synthesis modes and a .NET Framework4.8
runtime. `mono`, `dotnet`, and `mcs` were not found on PATH in the follow-up.
No runtime, compiled binary, example recordings, or participant measurements
were executed or fetched by the metadata discovery. Exact mode/configuration,
numerical parity, dependency compatibility, and small-data training protocol
remain unresolved. Source discovery is not a completed model reproduction.

The follow-up source acquisition then fetched and verified 18 C# files totaling
63,690 bytes, excluding examples, binaries and package directories. Local
`AverageSynthesizer` truncates the model mean to an integer and uses a random
0–999ms value when no context model exists; `LocalForwardSynthesizer` clamps
negative model-generated values to zero. `Program.cs` exposes several modes,
with `DEFAULT` routed to histogram synthesis. These implementation details
must be preserved or explicitly identified as adaptations; a deterministic
mean lookup alone is not automatically equivalent. The paper's chosen mode,
context configuration and seed still need verification. Receipts are in
`references/type2branch_synthesis/source_receipts.json`; the acquisition script
is `scripts/type2branch_synthesis_source.py` (`--sources` for the second stage).

## Source and exact requirements

Author repository commit:`6ddd9b0a13a0baf3f5631c73d26e2c2f67b08a1b`. Additional source files are retained in`references/type2branch/`, with the existing GPL license. [Official repository](https://github.com/lsia/tifs-type2branch).

| Contract | Author source |
|---|---|
| Architecture | Learned256×8 key embedding; temporal attention on each branch; two bidirectional256-unit GRUs and sequence self-attention;128/256/512 convolutions with kernel6 and channel attention;256D output. BN/dropout0.5 throughout. |
| Optimization | Adam0.0001; Set2Set loss margin1.5 and beta0.05. Loss uses non-squared pairwise Euclidean distances, ordered identity blocks, triangular within-user masking, and relative-radius regularization. It is not ordinary triplet loss. |
| Small configuration |15 samples per identity,10 identities per batch;100 training and100 validation steps per epoch; maximum600 epochs; patience40. Curriculum first changes at epoch20 and recomputes identity centroids. |
| Data | Author preprocessing selects up to15 samples per user and fixes100-event sequences. Its stock split requires1,000 validation plus1,000 evaluation identities, already exceeding all99 KeyRecs identities. |
| Extended features | HOWTO step2 invokes`merge_synth_features.py`, which requires externally produced CSV directories`xts/xvs/xes` and merges their non-leading columns. The17-file repository supplies a merger, but no producer for those files. |

The full maximum schedule is60,000 optimizer steps and9million sequence presentations, plus validation and centroid inference. Completing that maximum in25minutes would require40 optimizer steps/second before any validation or implementation work. No machine-specific Type2Branch training speed has been measured, so no runtime claim is made.

## Concrete blockers

1. The required`tensorflow`, `keras`, and`tensorflow_addons` modules are absent. The required bigidea environment is Python3.12. The official Addons compatibility matrix documents Python versions only through3.11 for its listed releases; its latest published0.23.0 wheels likewise target up toCPython3.11. A tested reference environment needs a compatible separate runtime or a verified replacement for the dependency. [Compatibility matrix](https://github.com/tensorflow/addons/blob/master/README.md), [published package files](https://pypi.org/project/tensorflow-addons/0.23.0/).
2. A PyTorch port would require numerical architecture/loss/gradient equivalence against the reference, including Keras GRU reset/bias conventions, attention normalization axes, BN behavior, and the source's specific Set2Set masking. A superficially similar model is insufficient evidence.
3. Full extended-feature reproduction needs validation of the now-located external generator, exact configuration and residual conversion. Silently omitting those channels would be an ablation. The earlier missing-producer assessment is superseded by the follow-up above.
4. Existing KeyRecs train blocks require a new explicit15-sequence identity eligibility/sampling contract and train-only validation adaptation. The published1,000-user validation protocol cannot be reproduced with99 total identities. A shorter schedule on this population would remain small-data budget transfer, even with identical architecture/loss.

## Work required before a faithful run

Obtain a compatible isolated reference runtime; resolve and validate extended-feature generation; validate a strict train-only KeyRecs adapter; compare reference and any port on synthetic forward/loss/gradient fixtures; benchmark one complete training epoch on the actual CPU; then preregister a justified training budget and stopping rule. Preserve all existing artifact versions and use dev only after freezing. A GPU or longer schedule may make training practical, but requirements cannot be quantified without the epoch benchmark. No pretrained checkpoint was present in the audited official tree or releases.

Related: [[keystroke_sources]], [[keystroke_worklog]].

## Executable input audit and generator defaults

The generated-only audit in `type2branch_input_audit_v1/report.json` completed.
It executes the author's extracted preprocessing function: keys divide by255,
millisecond timings divide by1000, sequences pad to100 rows, and timings clip
at30 seconds. The paper describes10 seconds, leaving a fidelity discrepancy.
The generator CSV reader instead skips a header and parses integer VK/HT/FT.
A structural emulation rejects all99 remaining rows of the normalized100-row
CSV; a header plus two integer rows accepts both. This is not .NET runtime
parity. Supply and validate an explicit units/header/padding bridge before use.

Pinned `configuration/App.config` and `packages.config` were acquired with
publisher Git blob/size checks and local SHA256 receipts (9,128 bytes total).
All four model sets specify maximum context7 and n-gram1, using memory storage
for HT and FT. `RNG.cs` seeds its global generator with1234, then obtains each
thread-local seed from global.Next(); Python seed1234 is not equivalent.
These are generator repository defaults, not proof of the paper's settings.
Mode selection, residual construction, runtime parity, and paper-versus-code
clipping remain unresolved. No dataset observations or accuracy measurements
were added by these audits.

## Strict bridge primitives and synthesized timing cleanup

`scripts/type2branch_csv_bridge.py` now provides strict integer CSV encoding,
exact output key/count alignment, raw observed-minus-synthetic millisecond
residual conversion to seconds, and padding after feature construction.
It rejects fractional values, overflow, malformed rows and event loss rather
than silently rounding or dropping data. Fourteen generated checks pass;
source snapshots and results are in `type2branch_bridge_v1/report.json`.
This is a wire-format component, not a full feature pipeline or runtime port.

Independent source review identified two postprocessing dependencies, now
acquired and hash-verified under `references/type2branch_synthesis/cleanup/`.
`ThresholdPartitioner` records positions whose synthesized FT exceeds1500ms.
`CleanFTs` maps negative HT/FT to1500ms, caps positive HT/FT at1500ms, then sets
first FT and partition-start FT to `int.MinValue` (-2147483648). These operations
follow synthesis. Therefore simply subtracting the exported integers would
produce invalid enormous residuals at sentinel positions. The raw subtraction
primitive intentionally exposes those values; it must not be used as final
model input until invalid-timing policy and reference semantics are resolved.
The emitter's exact sentinel is now established from source, while .NET runtime
parity and the paper's conversion procedure remain unverified. No real data was
read. Explicit true lengths and file-ID mapping are still required: padding
cannot be inferred safely from zero rows, and generated filenames use numeric
IDs rather than the Type2Branch U/S naming scheme.

## CPU architecture execution follow-up

A repository-local TensorFlow CPU2.16.1 / legacy tf-keras2.16.0 candidate now
executes the unchanged published model/configuration on generated inputs.
SciPy1.13.1 is pinned alongside NumPy1.26.4; the first failed attempt exposed
an incompatible SciPy imported from the other research dependency directory.
A second attempt exposed legacy Keras passing the integral float1e9 to
Python3.12 random.randint. A recorded one-literal dependency patch changes
that bound to integer1000000000. Original/patched hashes and original backend
source are retained; this is not the original author's runtime.

The third attempt completed:3,139,688parameters,57trainable tensors with finite
gradients, output2x256. Batch2 first-call forward1.151s and forward/backward
3.194s are diagnostic timings, not an epoch forecast or optimizer benchmark.
No Set2Set loss, optimizer update, dataset observations, or recognition metrics
were involved. `type2branch_reference_smoke_v1` and `v2` retain failed plans;
`v3/report.json` holds the successful evidence. The prior blanket statement
that TensorFlow/Keras are absent is superseded for this isolated candidate;
Addons and complete loss/runtime equivalence are still unresolved.

The public upstream discussion also confirms that the author corrected a
training-configuration issue and added the alternative configurations already
in our pinned source. It does not supply the missing residual-conversion code.
See [author response](https://github.com/lsia/tifs-type2branch/issues/2#issuecomment-2585771770)
and `type2branch_upstream_audit_v1` for four bounded, checksummed metadata
responses. Reported author accuracy is not a local benchmark result.

## Set2Set and full-cardinality optimizer execution

The official Addons v0.23.0 distance source is now pinned to commit
`6f07f87f8949519d79b0b2ab9599ffc0249f8a2b`, with Apache license and receipts.
Its selected original function and the selected original Type2Branch loss
class/function execute without installing the full Addons distribution.
No function bodies were replaced. The generated audit independently checks
unsquared Euclidean distance, three hand-calculated order-sensitive losses,
NumPy loss values, finite-difference gradients, and the published K10/N15
cardinality. Random loss error2.92e-7, maximum gradient error5.96e-8; full-size
loss1.5827059746 versus oracle1.5827061004. Collapsed embeddings yield nonfinite
loss/gradients in the source; this defect is preserved and reported.

A separate generated150x100x5 end-to-end optimizer step uses the unchanged
model, published configuration, original loss definitions, and author Adam
optimizer. It completed with finite loss2.4867742 and finite gradients/weights,
updating all57trainable tensors. First-traced step19.7335seconds, process peak
RSS2456.32MiB. This is not steady-state epoch throughput, convergence, dataset
validation or recognition performance. Artifacts are `type2branch_loss_audit_v1`
and `type2branch_optimizer_smoke_v1`. This narrows the Addons execution blocker;
full package/runtime equivalence and feature conversion remain unresolved.

## KeyRecs TRAIN count feasibility and inner roles

A source-pinned prefix-only audit counted223,078permitted session1digraph rows
across79active identities, reproducing the original manifest total. Timing/key
suffixes were never decoded. Full100-row windows per person range24–47.
The earlier50/25/25chronological candidate gives only11people with15fit windows
and zero with15selection windows; it cannot populate reference selection sets.
No participant was dropped to conceal this result.

All79people have at least24full windows over their entire TRAIN session.
A separate deterministic SHA256 identity partition therefore freezes47fit,
16selection and16calibration identities within the original TRAIN session.
Only fit identities may update global encoder/population parameters. Original
DEV/session2 and sealed test identities retain their roles. This is an explicit
small-data adaptation, not the paper's1000-user validation or direct parity
with older KeyRecs models trained on all79identities. Future DEV results must
separate representation-seen and representation-unseen cohorts.

Artifacts: `type2branch_keyrecs_eligibility_v1` and `type2branch_train_roles_v1`.
Three generated prefix/count/boundary tests pass. Exact window/support/probe
selection, digraph continuity, raw-key mapping and extended-feature conversion
remain to be frozen and validated before measurement decoding/training.
Counts alone do not establish event quality or independent sequences.

## Fitting-only digraph continuity evidence

The preregistered47-identity TRAIN-fitting scan decoded132,428rows, with source
and role hashes verified.57rows could not be decoded;2,114DD values were
nonpositive. All decoded within-row timing equations passed, as did key and
inferred-next-hold checks on130,205adjacencies. At the fixed1e-5second diagnostic
tolerance,30of47identities retain15candidate100-rowwindows when invalid rows
break runs. No cohort has been selected; all47remain in the diagnostic report.

This evidence is conditional on the conventional timing-column semantics;
source recorder resolution and absolute-event/task continuity remain unproven.
Zero versus negative DD and missing versus malformed rows need separate
classification before an actual input policy is fixed. Negative UD/UU are
allowed. Selection/calibration/DEV/test observations were not decoded.
Artifacts: `type2branch_continuity_v1`; three generated boundary/role tests pass.

## Invalid-row classification follow-up

A fitting-only follow-up reproduces all prior counts and separates2,111negative
DD intervals from3zero intervals. The57undecodable rows include10key CSV errors
and47empty-key records with188missing timing fields. These diagnostic categories
can overlap. All132,381complete finite timing rows are numerically compatible
with whole milliseconds within1e-6ms; no rounding was performed. Clock accuracy
and ordering semantics remain unverified. Artifacts `type2branch_invalid_rows_v1`
and three passing generated tests preserve the evidence without repairs,
cohort changes, or selection/calibration/DEV/test timing access.

## Fitting tail and ordering evidence

The follow-up `type2branch_order_audit_v1` establishes that all47incomplete
records are final rows, exactly one per fitting identity. No interior missing
record was observed. All2,116negativeDD complete-timing rows have nonnegativeUU;
one other row has negativeUU, and no inferred second hold is negative. These
sign patterns are consistent with overlapping events but prove neither strict
keypress nor strict release ordering. No records were reordered/repaired.
Two generated boundary tests pass; previous totals/source hashes reproduce.
Independent primary-source research did not locate an accessible collector
implementation or explicit ordering/terminal convention. A future KeyRecs
adapter must label any event-order choice as an adaptation and preserve these
observed limitations rather than invent author parity.

## Base-channel adapter now executed

`type2branch_fit_base_v1` prepares705sequences from all47fitting identities,
first15full100-rowwindows each. No identity dropped/substituted. It preserves
published row order as an explicit adaptation rather than rejecting every
negativeDD; source preprocessing itself clamps negative incoming timing to0.
Signed values and original boundary intervals remain in the audit arrays.
Known-key/hold adjacency is checked; unknown key prefixes map0 with flags.
Terminal current events are retained under an explicitly provisional pattern.
All1,302fitting windows match author prepare_sample arithmetic exactly; this
does not establish collector semantics. Four generatedtests pass. Selection,
calibration, DEV and test observation values were not decoded. The705x100x3
arrays lack two synthetic channels and are not a complete model input.

## Input cleanup and context contracts resolved from source

Seven additional pinned source files (33,862bytes) establish that Program's
LoadDataset cleans inputs before population fitting; cleanup is not confined
to synthesis output. It partitions FT>1500ms, maps negative timings to1500ms,
caps timings at1500ms, then marks first/partition FT int.MinValue. The default
context selector requires10observations and chooses the longest qualifying
context. Context starts with0xFF; MemoryStorage skips hash0 during bulk lookup.
These rules rule out a naive unrestricted mean lookup or feeding signed CSV
values directly into timing models.

Source-translated cleanup now has17combined generated bridge/cleanup checks
passing. On705fitting windows only, it produces678pause positions and1379invalid
FT positions;953negative inputFT are handled by the reference cleanup rules.
Artifact `type2branch_synthesis_fit_v1` preserves cleaned values and partition
masks separately from original signed/base arrays. No population was fitted;
C# runtime/RNG parity, synthesis mode and residual conversion remain pending.

## Population mean-context candidate fitted

`type2branch_context_fit_v1` now fits164,156HT/FT context models on705frozen
sequences from47fit identities only. Nine generated tests pass, including
FFinitial history, ordered partition resets, minimum10observations, longest
context backoff, separate order/hash storage and hash0 lookup exclusion.
Input domain0–1500ms avoids the source's int32 square-overflow hazard. Source
and serialized artifact hashes are verified; fit took2.124seconds.

Unpartitioned dummy-key queries on the same fitting sequences leave61positions
per timing channel without a qualifying model. Missing values remain explicit;
source RNG fallback has not been substituted. This source-translated Average
candidate is not verified paper mode (CLI default is Histogram) or C# runtime
parity. No residual channels, encoder optimization or DEVmetrics follow from
this population-only result. Selection/calibration/DEV/test values were not read.

### Seeded fallback audit

`type2branch_random_audit_v1` validates the legacy random translation against 200 Microsoft seeded Next values and derived double values. Thirteen combined RNG/context tests pass. The Average candidate now has a tested missing-context filler with persistent first-thread stream and HT-before-FT traversal. This is source/vector validation, not full CLR parity; actual fit synthesis, residual sentinel handling, and encoder fitting remain pending.

### Actual five-channel fitting candidate

`type2branch_synthetic_fit_v1` generated all705frozen fitting sequences using the Average candidate (122fallbackdraws). Cleanup invalidates only firstFT,705positions. `type2branch_fit_five_channels_v1` contains finite705x100x5inputs: unchanged authorbase plus signed residual seconds, zero residual at missing synthetic timings with separate validity mask. This residual policy is explicitly adapted because the publisher merger does not define it. Nineteen combined tests pass; source/input/output hashes are recorded. No other-role observations, encoder training, or accuracy metrics yet.

### Inner-TRAIN roles prepared

`type2branch_inner_train_features_v1` supplies240x100x5sequences each for16selection and16calibration identities, all originalTRAIN/session1. Population stays fit47-only; first5windows/gallery andlast10/probes frozen. No metrics or gradients computed from these roles yet. DEV/test observations remain unread by this stage. Actual source training assumes exactly15samples/identity; curriculum timing must preserve100updates/epoch and20epoch delay or disclose an adaptation.

### Actual outcome: trained but frozen DEV infeasible

Initial300updates completed; trainedepoch2selected. CalibrationFAR1%,FRR75.625%,pooleddiscreteEER20.7083%. All79enrollmentgalleries prepared; p004S2has38rows, belowrequired1000. Complete-cohortgate leavesDEVmetrics empty; no droppedidentity orchangedwindows. ActualfailureAPI replay passes, not biometricpredictionparity. This is an incomplete/negative experiment, not SOTAperformance.

### TRAIN-only length sensitivity

Following the disclosedDEVcountfailure, `type2branch_length_audit_v1` tested onlyinnerTRAINselection probes withfullgalleryfixed. PooleddiscreteEER worsensfrom14.9167%(100events)to18.2083%(75),30.625%(50),45.625%(25). No minimumlength/thresholdchosen; this confoundsreducedinformationwithunmaskedpadding andisnot true shortenedrawsynthesisparity. Ten generatedtests pass. FailedDEVv1remainsunchanged;shortprobesneedTRAIN-onlyrobustnessworkbeforefollow-upvalidation.
