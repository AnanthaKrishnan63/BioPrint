# Objective completion audit

The full goal remains **incomplete**. This audit distinguishes verified progress
from unresolved requirements; it does not redefine the objective around the
available datasets.

| Requirement | Current evidence | Status |
|---|---|---|
| Newer fixed and free-text typing data | KeyRecs2023 acquisition, immutable checksums, frozen fixed/free reports; TypeNet transfer | Available-data work complete; free task is transcription |
| Pointer, device, other feature data | Balabit, SapiMouse, FPStalker, DELBOT, cognitive tasks, TSI, BEACON manifests and source notes | Broad coverage; exact keypad and several live features still missing |
| Each dataset under 500 MB; total under 5 GB | `dataset_budget_paired_type2branch_complete.json`: 1,211,692,106 bytes; conservative largest-source bound 468,685,321 bytes; source documents included | Verified at recorded snapshot within declared scope |
| Separate train/dev/test; never inspect test | Split-aware loaders, metadata role ledgers, sealed-route/poison-record tests, download receipts | Current pipeline guards verified; historical CMU exposure explicitly prevents pristine-test claims |
| Train-only fitting and optimization | Frozen configuration files and training-selection records for each experiment; dev API reads only frozen models | Implemented for recorded experiments; dev must never become a tuning loop |
| SOTA baseline for every feature | Primary-source research, TypeNet and author FCN architecture work, practical tree/SVM references | Incomplete: not every method is SOTA; exact source protocols, weights and task data unavailable in several areas |
| FAR/FRR/EER and improvement | `summary_v11.json`: 72 rows, 25 source reports; three infeasible experiments including fixed-capture Type2Branch | Verified measurements, not universal improvement or near-zero errors |
| Type2Branch short-capture follow-up | Completed paired TRAIN selection/calibration; DEV79/79coverage, FAR9.13664%, FRR30.37975%, EER13.93216%; actual API6241scores bit-exact, zero decision flips | Executed adapted baseline; target FAR did not transfer; original failed protocol and prior exposure preserved |
| Type2Branch paired BEACON transfer | Eight frozen comparators; three DEV people, 243 claims, 140 windows including 81 below 25 events; hybrid FAR 24.69%, FRR 67.90%, EER 40.74% versus matched old hybrid 45.68%, 30.86%, 38.27%; actual API 1,944 scores bit-exact, zero decision flips at three thresholds, exact pooled/per-person metrics | Negative tradeoff; prior DEV exposure; offline encoders and fusion API boundary; no all-feature completion claim |
| Optimize every existing feature | Feature-specific experiments and `coverage_gaps.md` | Incomplete: exact randomized-keypad cognitive/motor and all browser/bot features lack representative validation data |
| Joint optimization and validation using all features | BEACON actual keyboard/mouse/context alignment, handcrafted, two neural fusion experiments, and a training-only nonlinear search retaining LR; API parity | Incomplete: genuine three-channel evidence only; no jointly labeled keypad/bot/all-device corpus |
| Scripts and real website APIs for dev validation | Isolated research app, dataset/sample endpoints, frozen scoring, exact replay reports; production CMU replay in temporary DB | Verified for available data; research API is not a silently deployed production replacement |
| Hardware-conscious small local models | CPU thread controls, model parameter counts, inference timing, observation-cost and exact-parity runtime optimization reports | Verified for implemented models |
| Ethical results and restricted writes | No artificial cross-dataset identities; failed reports preserved; no listener/live DB changes. One agent wrote a helper under `/tmp`; earlier direct pytest temporary-file locality was not assured. Subsequent runs use the repository wrapper. | Write restriction was violated and documented; later workflow corrected, no retroactive compliance claim |
| Master and experiment logs | `../../logs/WORKLOG.md`, `../../logs/EXPERIMENTS.md`, per-agent source/worklogs | Maintained; runtime/bot verification complete |
| Judging criteria and deliverables | Source brief reviewed; `REPORT.md`, website, setup, coverage audit; actual Git history at `a874140` | Explanation and existing history verified; final demo/submission packaging remains separate |

## Remaining decisive evidence

The later arithmetic task directly supplies three ordinal difficulty levels
and measured response timestamps. Eleven TRAIN participants supported a frozen
eight-candidate comparison selecting the full wait/missing/error profile with
Manhattan distance; calibration FRR91.67% at observed FAR0/36 is poor. The
four-person DEV attempt is infeasible under the unchanged parser: one required
enrollment block has50 entries rather than5. No DEV rates or API score replay
exist for that comparison. `arithmetic_train_v1/README.md` and
`arithmetic_dev_failure_v1/report.json` preserve this failure; the new source
does not resolve complete-cohort cognitive or all-feature verification.
The arithmetic status API has now been replayed in process: frozen reports
match exactly, DEV samples/scoring return409, and sealed/remote requests return403.
This is status/error integration evidence with zero biometric predictions.

An exact all-feature claim requires public records that jointly identify the
same actual participants across the implemented login, pointer, device, keypad,
and labeled attack flows. It also requires adequate repeated sessions and
calibration data. No acquired corpus supplies that evidence. Arbitrarily joining
users from different datasets, synthesizing missing fields, or calling unit tests
real biometric validation would not satisfy the requirement.

The present experiments support a reviewable research framework and several
measured improvements. They do not establish that every feature has been
optimized, that all-feature fusion works, or that false accept/reject rates are
near zero. Continue remaining meaningful verification and implementation work;
do not mark the full goal complete based on this partial evidence.

## Latest source and integration checks

`type2branch_feasibility.md` preserves the initial generator/runtime gaps and
the subsequently implemented, explicitly adapted synthesis and CPU runtime.
Actual Type2Branch training, DEV inference and API verification now exist; exact
published-protocol reproduction remains unproven. `pointer_lt_mouse_audit.md`
still documents unavailable architecture/training specifications. Neither
adapted execution nor an unavailable reproduction establishes universal SOTA. `beacon_nonlinear/README.md` records
a failed training-selected improvement and exact reference decision parity.
`bot_rules/api_replay.json` verifies all 5,100 genuine dev records through the
API, but cannot establish bot-positive FAR/EER or browser-trust validity.

The CMU per-account follow-up retains the enrollment and feature budgets,
reduces observed dev FRR at the frozen 1% FAR target, and passes all 780,300
three-model API score comparisons with zero changed decisions. Its EER delta
interval includes zero. This advances typing optimization but does not resolve
the missing all-feature corpus or exact SOTA reproduction requirements.

A further preregistered CMU selector optimizes training FRR at 1% FAR rather
than EER. Dev FRR is 64.10% at actual FAR 0.99%, while EER worsens to 18.50%.
It demonstrates an operating-point tradeoff, not improvement on every metric
or completion of the feature coverage requirements.

Further paired-data discovery is recorded in `bbmas_hmog_feasibility.md`.
HMOG selective range acquisition is now verified: 12 preregistered subject
archives are retained under the 500 MB source limit, with test identities
unacquired. Session roles are frozen; four fit identities' TRAIN sessions have
passed schema, activity-membership and duplicate-stream audits. No other cohort
was inspected during schema work. Strict extraction v2 yielded zero eligible
windows out of332 and remains preserved. Separately frozen gap-filtered v3
yielded207windows;87TRAIN-fit windows trained800updates. Separate TRAIN selection
chose epoch3. Original calibration failed; a separately frozen seen-identity,
disjoint-session fallback supplied thresholds. DEV full-cohort validation remains
infeasible because two of four participants have no eligible probes. The31
observed probes cover20.13% of154 inspected candidate windows. Joint EER45.16%,
FAR0%/FRR100% at the1% TRAIN target are conditional, negative results. Observed-
window API replay passed372 scores with zero decision changes while preserving
infeasible status and missing coverage. No model/threshold was changed using DEV.
This is executed partial evidence, not successful complete-cohort validation.
BB-MAS acquisition remains unverified. Neither source supplies every implemented
signal; acquisition alone is not model validation.

The subsequent touch follow-up adds genuine tap measurements to all207 existing fit windows. Frozen full11 and fixed3 TRAIN diagnostics have EER41.00% and42.53%, both FRR98.85% at0% observed FAR using1% calibrationtarget. These negative TRAIN results do not improve the DEV completion status, establish modern touch SOTA, or resolve all-feature joint validation. No new test or DEV observations were inspected.

Frozen six-method TRAIN-selection transfer failed to find useful discrimination: tap11 alone met zeroobservedFA but rejected45/45genuine probes; every othermethod exceeded1%observedFAR. All60originalwindows andbothgalleries available, so this failure is not missingcoverage. No globalparameter/threshold changed andno newDEVread. User-required DEV comparison for this experiment remains outstanding; anyfollow-up must preserve failedselection andearlierDEVexposure, without promotion.

The touch-augmented DEV experiment now has an actual frozen validation attempt: all68originalwindows yielded taps, but556357has77supporttaps versus80required. Four-accountcomparison isinfeasible beforeinference; metrics unavailable, notzero. No touchcomparison APIscorereplayexists. This negativeenrollment evidence doesnotcompleteall-featurevalidation orSOTAoptimization. NoDEV-basedrelaxation or testaccess occurred.

The subsequent SapiMouse enrollment-only Z-normalization candidate reduced TRAIN-calibration EER but worsened the prespecified low-FAR FRR objective (48.57%→54.29%). It was not selected; the existing cosine model, thresholds and earlier DEV/API evidence remain unchanged. No new DEV/test reads or new validation success are claimed.

The subsequent frozen Z-norm comparison now has DEV/API evidence: EER11.73%, FAR0.74%, FRR60.53%, versus unchanged cosine13.87%/1.66%/43.42%. This is a tradeoff and doesnotreverse TRAINselection. Actual3648scoreAPIreplay isbit-exact atallthreefrozenthresholds. It doesnotresolveall-featurefusion orrepresentativekeypad/botdata gaps.
