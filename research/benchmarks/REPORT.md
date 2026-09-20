# BioPrint research validation report

Status: ongoing experiments, not a completed production-security evaluation.

The consolidated [summary](summary_v11.json) contains 72 measured DEV rows from 25 frozen reports and
three separately recorded infeasible experiments. Missing metrics are not zeros.
The [completion audit](completion_audit.md) tracks the full requested scope;
current evidence does not establish SOTA baselines for every feature or
all-feature joint verification.

## Problem and implementation

The latest executed Type2Branch short-capture follow-up covers all79active
accounts, with **FAR9.13664%, FRR30.37975%, pooled discrete EER13.93216%**.
The model and exact-length thresholds were selected using TRAIN only; the1%
calibration FAR target did not transfer across sessions. Actual API inference
matched all6241offline scores and every reported metric exactly. This is a
documented adaptation with prior DEV exposure and one probe per identity,
not an exact paper reproduction or all-feature fusion result. The original
failed1000-event protocol remains recorded separately.

The subsequent [paired BEACON transfer](beacon_type2branch_dev_evaluation_v1/report.json)
evaluates eight frozen comparators on three people and243claims (81genuine,
162impostor). All140paired windows remain, including81below the source model's
25-event training minimum. At the frozen1% TRAIN FAR target, the Type2Branch
hybrid yields **FAR24.69%, FRR67.90%, EER40.74%**, compared with45.68%,30.86%,38.27%
for the matched old hybrid selected under the same new TRAIN rule. This is a
negative overall result: fewer false acceptances trade against substantially
more false rejections and worse EER. Neither calibration target transfers.
Encoders stay frozen; only fusion models fit on BEACON TRAIN. Earlier DEV
exposure remains disclosed. [Actual in-process API replay](beacon_type2branch_api_replay_v1/report.json)
reproduced 1,944 scores (eight models × 243 claims) bit-exactly, all decisions at three
thresholds, and every pooled/per-person metric. Access and malformed-request
guards passed. API scope is frozen fusion on offline encoder features, not a
live capture-to-encoder pipeline or all-feature verification; no listener started.

BioPrint examines whether browser behavior is consistent with an enrolled
account holder. Typing rhythm, pointer movement, cognitive tasks, device context,
and bot indicators answer different questions. A matching browser fingerprint
does not prove personal identity. Missing or conflicting evidence should trigger
additional verification. Research model improvements are not automatically
installed into existing users' profiles.

The live typing baseline compares hold/interkey timings against enrollment
medians and dispersion. Research comparisons include class-balanced RBF SVMs,
tree ensembles, a 200,458-parameter TypeNet architecture transfer, browser-pair
classifiers, pointer classifiers, and separately trained paired-modality fusion.
Published architecture implementations are distinguished from exact published
benchmark reproductions. CPU training budgets and task/domain changes prevent
claiming equivalence to published SOTA numbers.

## Experimental controls

Each acquisition has training, validation/dev, and sealed test assignments.
Fitting, model selection, and threshold calibration use training partitions only.
Unseen-user first-session enrollment is explicitly a training-support role;
second-session dev probes never fit templates or global models. No unrelated
datasets are joined through artificial identity mappings. Earlier legacy CMU
exploration prevents a retroactive pristine-test claim; current loaders reject
sealed data access and ordinary unit tests use synthetic fixtures.

FAR is the fraction of impostor comparisons accepted; FRR is the fraction of
genuine comparisons rejected. Diagnostic dev EER summarizes a score sweep;
operational FAR/FRR use thresholds frozen on training calibration. A target
training FAR is not a promise about the achieved dev FAR.

## Current frozen dev evidence

These rows describe different tasks and protocols and must not be ranked against
one another. Percentages are rounded. FAR/FRR use the training 1% FAR target.
CMU/KeyRecs EERs are per-account averages; other rows use pooled EER. On SapiMouse,
the author OCSVM's macro EER is 7.73%, versus 8.27% for selected cosine scoring;
cosine improves pooled EER and the frozen global operating point, not every metric.

| Dataset / method | Dev EER | Actual FAR | FRR |
|---|---:|---:|---:|
| CMU, ten-enrollment live baseline | 22.04% | 1.10% | 74.20% |
| CMU, matched-enrollment global RBF SVM | 15.50% | 1.00% | 72.98% |
| CMU, matched-enrollment per-account RBF SVM | 14.32% | 0.95% | 67.43% |
| CMU, per-account selection for FRR at 1% FAR | 18.50% | 0.99% | 64.10% |
| KeyRecs fixed, ExtraTrees | 11.62% | 1.29% | 45.91% |
| KeyRecs free transcription, ExtraTrees | 9.72% | 1.14% | 45.64% |
| KeyRecs free, TypeNet architecture transfer | 20.23% | 1.17% | 84.08% |
| FPStalker browser linkage, gradient boosting | 2.30% | 0.84% | 4.85% |
| DELBOT human/bot geometry, random forest | 14.55% | 3.64% | 31.15% |
| Stroop/Flanker, full timing/accuracy profile | 27.78% | 2.78% | 77.78% |
| Touch TSI, target-normalized motor/timing RF | 34.45% | 1.00% | 91.85% |
| SapiMouse, five-block handcrafted baseline | 26.17% | 8.30% | 60.53% |
| SapiMouse, five-block FCN/cosine profile | 13.87% | 1.66% | 43.42% |
| SapiMouse, unselected Z-normalized cosine comparison | 11.73% | 0.74% | 60.53% |
| BEACON paired keyboard/mouse fusion | 44.14% | 2.47% | 98.77% |

Detailed results and uncertainty are in individual JSON artifacts and
[the experiment log](../../logs/EXPERIMENTS.md). Balabit pointer results were
mixed: selected-model macro EER improved while pooled EER worsened. SapiMouse
improved at matched five-block observation cost, but those blocks require 641
coordinates and a training median of 16.89 seconds. Both paired representation-transfer experiments have finished without a
reliable improvement. A subsequent training-only nonlinear search retained the
original logistic regression; rejected nonlinear candidates were not dev-scored.
Negative results are retained without dev-driven retuning.

The Z-normalization candidate uses TRAIN-derived background statistics and
unchanged observation cost. Its lower DEV FAR/EER accompanies higher FRR;
the original TRAIN selection retains cosine. A paired 2,000-replicate
probe-person bootstrap gives an EER-difference interval of −5.47 to +3.98
percentage points. This interval conditions on fixed enrollment references
and models and does not account for prior DEV exposure. The apparent EER
gain is uncertain. See [comparison](pointer_sapimouse_znorm_dev_v2/README.md)
and [uncertainty](pointer_znorm_uncertainty_v1/README.md).

## What the evidence does and does not show

The CMU supervised reference improves matched-budget EER, but stringent-FAR
false rejections remain unacceptable for frictionless login. More enrollment
and repeated observations change the user-effort budget and are reported as
separate comparisons. TypeNet's smaller local training run did not beat the
feature-based free-text model.

Stroop/Flanker slopes barely distinguished nine dev accounts; richer profiles
performed better but required hundreds of trials. They do not validate a
three-question arithmetic challenge or scrambled keypad. Touch TSI adds real
target-relative landing evidence, with poor accuracy in one-visit validation;
it supplies neither release timing nor randomized-keypad reaction times.
BEACON includes only
three dev people playing games, and fusion did not improve its first validation.
It provides no jointly measured keypad/bot evidence. Device results measure
browser linkage, with feature availability differing from the live collector.
Neither zero-error authentication nor full-feature production reliability has
been established.

A later author-released arithmetic dataset directly provides three ordered
difficulty levels. Its complete release has 19 participant archives despite
20 described in the paper. The frozen split uses 11 TRAIN, four DEV, and four
unacquired test identities. Eight TRAIN candidates compare mean waiting time,
ordinal slope, condition means, and timing/missing/error profiles with two
distance metrics. Missing responses remain distinct from measured reaction
times; bounded waiting time includes the 2.5-second deadline.

TRAIN selection chose the full profile with Manhattan distance. Final TRAIN
calibration rejected 11/12 genuine probes at 0/36 observed impostor accepts;
those rates are not DEV performance. DEV enrollment failed because one required
block has 50 entries rather than the frozen five-trial contract. The other59
blocks pass, but no participant removal, truncation or subset scoring was used.
DEV recognition metrics are unavailable. The API reproduces this failure status
and refuses scoring; it does not provide biometric prediction validation.
See [arithmetic report](arithmetic_train_v1/README.md).

## Reproducibility and demonstration

[Research commands](README.md) run through the `bigidea` environment, with local
CPU dependencies and caches. The isolated research website presents frozen
reports and exposes bounded feature/score APIs. Scripts replay dev inputs via
in-process ASGI without a network listener or access to `bioprint.db`. Score and
decision parity are checked against frozen offline models. Production API
registration/enrollment/login was separately exercised with public CMU records
in a temporary database; that sparse integration sample is not the full benchmark.

For judging, reliability is supported by reproducible FAR/FRR reporting and
disclosed failures; innovation by separated behavioral/context evidence and
explanations. User effort, observation duration, and inference latency must be
measured rather than inferred from accuracy. The runnable demo, documentation,
and existing Git history are present (current HEAD `a874140`). Research edits
remain working-tree changes; final submission packaging is separate. No stronger
accuracy claim is warranted.

Compute-only timing is recorded in `inference_latency_singlethread.json`: selected
SapiMouse inference took a median 6.8 ms for 24 claims, while collected movement
took seconds. Removing tree-worker overhead reduced KeyRecs 79-claim inference
from roughly 65–71 ms to 13 ms, with every frozen decision unchanged. These warm
measurements exclude network, capture, and cold loading.
An initial timing report failed to await async pointer inference and is retained
with an explicit invalid filename; it is not evidence.

The bot-rule API reproduced scores, flags, and ordered rule explanations for
all 5,100 genuine CMU dev records. It flagged 15 records (0.294%), all under the
configured short-hold heuristic. This is not independent validation of the
historically CMU-informed rule or an estimate of bot FAR/EER: the source lacks
bot-positive labels and browser-trust telemetry.

Audits of Type2Branch and LTMouseAuthen found unresolved reference-runtime,
feature-generation, or architecture-specification gaps. See
[type2branch_feasibility.md](type2branch_feasibility.md) and
[pointer_lt_mouse_audit.md](pointer_lt_mouse_audit.md). Those methods have not
been faithfully reproduced here.

A later source audit located Type2Branch's paper-linked external synthesis
implementation and pinned its C# sources. This removes the assumption that the
generator is unavailable, but exact configuration, residual-feature conversion,
runtime compatibility and numerical parity still need verification. No new
Type2Branch model or accuracy result is implied by source acquisition.

A subsequent fixed-grid CMU experiment selected SVM parameters separately for
each account using training sessions 2–3 and calibrated thresholds on session 4.
At the 1% training FAR target, dev FRR improved by 5.55 percentage points
versus the frozen global SVM; the paired-account bootstrap interval is
−10.49 to −1.43 points. EER changed by −1.18 points, with interval
−2.56 to +0.26, so its apparent improvement is uncertain. Shared impostor
probes limit the account-bootstrap interpretation; previously exposed dev data
make this exploratory follow-up. No feature, threshold, or model was selected
from those dev outcomes. The models remain research-only.

A separately preregistered selector minimizes training FRR at 1% FAR, using
EER only to break ties. On dev, FRR falls another 3.33 percentage points
versus EER-based per-account selection (paired-account interval −6.96 to
−0.37), with actual FAR increasing from 0.949% to 0.985%. EER worsens from
14.32% to 18.50%, and FRR at the 5% training FAR target also worsens. These
are alternative tradeoffs; no dev-based choice between them is promoted.
The training objective, candidate grid and calibration rules were frozen first.

A bounded HMOG acquisition now supplies 12 preregistered participants with
repeated paired mobile recording archives. The approximately 428.5 MB subset
was selected by archive size, creating a completeness/recording-length bias.
Identity partitions precede acquisition; session roles are now frozen and test
identities were not downloaded. Guarded schema audits cover only four fit
identities' TRAIN sessions. Separate TRAIN-selection and calibration cohorts
were processed under frozen stage guards. DEV validation is complete as an
observed-window experiment; test observations remain sealed.
The audit confirms keyboard-touch anchors and detects duplicate
touch streams; it does not establish recognition accuracy. Pure tap features and
the author BehaveFormer architecture have synthetic checks. A permissive first
audit found 208/332 eligible windows, but a stricter contact/contiguous-event
contract yielded zero. A separately frozen retained-pair adaptation yields207
windows, explicitly reporting omitted-event gaps. Its87fit windows trained the
author architecture for800updates in336.67seconds, peak565MiB. Separate TRAIN
selection chose epoch3 (joint EER35.56%,45genuine/45impostor scores); this is
optimistic selection evidence, not DEV performance. Original identity-disjoint
calibration was infeasible; a separately frozen fallback uses disjoint enrollment
sessions from encoder-seen identities. Its optimistic-transfer risk is explicit.
The strict extraction and original calibration failures remain preserved. HMOG is
restricted to noncommercial research/education and must not be redistributed;
the College of William and Mary bears no responsibility for our analysis or
interpretation. See `bbmas_hmog_feasibility.md` for the official source and terms.

HMOG full-cohort validation is **infeasible**: two of four DEV participants have
no eligible probe windows. All four have galleries;31 observed probes from two
participants cover31/154 inspected candidate windows (20.13%). Other metadata-
excluded sessions have unknown candidate counts. At the frozen1% training FAR
target, conditional observed results are:

| Same-checkpoint output | Pooled EER | Actual FAR | FRR |
| --- | ---: | ---: | ---: |
| Joint typing+IMU | 45.16% | 0.00% | 100.00% |
| Keyboard ablation | 26.88% | 0.00% | 93.55% |
| IMU ablation | 50.54% | 4.30% | 96.77% |

These negative, coverage-limited results support no deployment or improvement
claim. The ablations share joint training and checkpoint selection. An explicit
partial replay release retains infeasible status, missing probes and all four
galleries. Actual API replay verified372 scores across three outputs, maximum
absolute difference3.22e-6, zero acceptance flips and identical pooled metrics.
It started no listener, used no production database and rejected the test route.
See `hmog/README.md`, `hmog/validation_v1/validation_complete.json` and
`hmog/api_replay_v1/replay_complete.json`.

The later HMOG touch-augmented comparison also failed complete-cohort enrollment:
one account has77 support taps below the frozen80-tap minimum. The minimum was
not lowered using DEV. Its empty metric report must not be confused with the
earlier keyboard/IMU observed-window results. See
`hmog/tap_dev_v1/dev_complete.json`.

## Type2Branch source-backed candidate: trained, DEV infeasible

The3,139,688-parameter author model/loss/generator completed300CPUupdates in26.8minutes(~2.04GiBpeakRSS). InnerTRAINselection choseepoch2(loss0.37366). Explicit source-order, Average synthesis and missing-residual adaptations are documented; threeepochs do not reach the source curriculum or establish convergence.

InnerTRAINcalibration at frozen1%FAR gives75.625%FRR and20.7083%pooleddiscreteEER(15.6667%macro). These are poor calibration results, not DEV performance. The unchangedcross-sessionDEV protocol is infeasible:p004S2has38rows versus1000required; all79enrollment galleries succeeded, but no subset was scored. Recognitionmetrics remain unavailable.

ActualASGI failure replay returnsstatus200,DEV/scoring409,train/testandremote403,withzeropredictions. Actualmodelworkerreplay remains unverified. `summary_v9.json` retains63DEVmetricrows and records threeinfeasibleexperiments separately; no Type2Branchcalibrationmetric is inserted asDEV. See type2branch_train_v1, type2branch_calibration_v1, type2branch_dev_features_v1 and type2branch_failure_api_v1.
