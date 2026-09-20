# HMOG paired typing and motion experiment

This experiment uses the pinned BehaveFormer IJCB2023 author architecture for
paired typing, accelerometer and gyroscope observations. It is a preprocessing
and small-budget adaptation. It does not establish published-performance
reproduction, current SOTA, or coverage of every BioPrint feature.

## Data and observation contract

Twelve participants were selected by archive size before acquisition, introducing
recording-length and completeness bias. Immutable identity roles assign four
fit, two selection, two original-calibration and four DEV participants. Sessions
17 and above remain sealed test data; reserved test identities were not fetched.
The role guard checks archive, member and upstream-artifact provenance before
opening observations. Missing/empty keypress sessions retain unknown candidate
window counts in coverage reports.

Each eligible window spans 30 seconds on recorded absolute time within one
writing activity. Fifty keyboard tokens use 52 complete, uniquely touch-matched
key pairs. Incomplete or ambiguous events are omitted without synthesizing
endpoints. Consequently, timing channels describe adjacent **retained pairs**
and may span omitted physical keys. Gap distributions accompany extraction.
The 100 sensor bins require actual accelerometer and gyroscope support, portrait
orientation and valid recording-clock order. Source FFT and gradient operators
are applied within the window before binning. This is recording-time association;
hardware synchronization is not established.

## Training, selection and calibration

The 3,376,714-parameter author encoder trained on 87 fit windows from sessions
9–16. The fixed schedule used 800 updates, eight cross-session triplets per
update, three separate forward calls, the exact source loss, Adam0.001 and two
CPU threads. Training took336.67seconds and peaked at about565MiB RSS.
The 120 fit enrollment windows did not enter encoder optimization.

Separate TRAIN-selection chose epoch3 from20 fixed checkpoints by minimum pooled
joint EER, breaking ties by earliest epoch. Selection EER35.56% is optimistic
selection evidence, based on45 genuine and45 impostor claims. Keyboard and IMU
outputs are ablations of that same jointly trained and selected checkpoint.

Original identity-disjoint calibration failed because one assigned participant
had no eligible windows. A separately frozen fallback uses the first eligible
numeric enrollment session per fit identity for its gallery and remaining
enrollment sessions for calibration:37 gallery windows,83 genuine and249 impostor
claims. Those observations were not optimized, but the encoder has seen their
identities. Thresholds may therefore be optimistic for new identities. The
fallback policy, failure and limitation remain bound into downstream reports.

## Evidence and limitations

- `fit_feature_v2_results.json`: strict contiguous-event contract, zero usable
  windows out of332; preserved unchanged.
- `fit_feature_v3_results.json`: separately frozen retained-pair adaptation,
  207/332 windows; median endpoint retention81.25%.
- `training_v1/`, `selection_v1/`: training history and exact checkpoint choice.
- `calibration_v1/`: original infeasible calibration.
- `calibration_fallback_v1/`: seen-identity threshold calibration.
- `model_protocol_v2.json`, `calibration_fallback_protocol_v1.json`: frozen rules.

DEV validation retains infeasible full-cohort status: two accounts have no
eligible probes, although all four have galleries. The31 observed probes cover
20.13% of154 inspected candidate windows. At the frozen1% TRAIN FAR target:

| Same-checkpoint output | Observed pooled EER | Actual FAR | FRR |
| --- | ---: | ---: | ---: |
| Joint | 45.16% | 0.00% | 100.00% |
| Keyboard ablation | 26.88% | 0.00% | 93.55% |
| IMU ablation | 50.54% | 4.30% | 96.77% |

These are poor, conditional results from only two probe identities. Missing
windows and accounts remain explicit; zero observed FAR with100%FRR is not
successful authentication. No DEV-based tuning or production promotion occurred.
The small cohort and dependent impostor comparisons cannot establish population
low-FAR guarantees.

Run scripts through `bash scripts/research.sh` to use the bigidea environment and
repository-local temporary/cache directories. Frozen outputs refuse overwrite.
Generated checks are in `scripts/test_hmog_*.py` and
`code/bioprint/tests/test_research_hmog_api.py`. An explicitly hash-pinned partial
research release exposes only the observed DEV probes and all four galleries,
preserving infeasible status and missing coverage. Actual API replay verified372
scalar scores, maximum difference3.22e-6 and zero acceptance flips; it started no
listener and did not import the production database. See
`api_replay_v1/replay_complete.json` and `metadata_erratum.md` for the preserved
release's calibration-scope metadata omission.

HMOG is restricted to noncommercial research/education and must not be
redistributed. The College of William and Mary bears no responsibility for this
analysis. Source and license provenance are in `../references/hmog/` and
`../hmog_schema_sources.md`.

## Touch reference follow-up: TRAIN only

A separately frozen extraction retains all207 original paired fit windows and
17,696 valid tap vectors. This is coverage within the existing key/IMU-selected
windows, not all raw-session windows. First eligible enrollment session per
account supplies37 gallery windows; remaining enrollment sessions supply83
calibration windows; sessions9–16 supply87 TRAIN diagnostic windows.

| Touch reference | TRAIN diagnostic EER | FAR at calibrated1% target | FRR |
| --- | ---: | ---: | ---: |
| Full11 feature family | 41.00% | 0.00% | 98.85% |
| Fixed published3 subset | 42.53% | 0.00% | 98.85% |

Both are negative results. The fixed3 subset uses duration, mean contact size
and inter-press velocity; it is not a rerun of the paper's mRMR selection or a
modern SOTA reproduction. Numerical conventions and the five-tap enrollment
window restriction are explicit adaptations. Neither diagnostic accessed DEV
or test observations. Original encoder DEV exposure remains part of the record.

Artifacts: `tap_train_protocol_v1.json`, `tap_train_v1/`,
`tap_three_train_v1/`, and `fit_tap_extraction_results_v1.json`.
The existing encoder trained on these87 diagnostic observations, so any paired
encoder comparison here must be labeled optimistic TRAIN evidence.

## Paired touch/typing/motion TRAIN diagnostic

All methods use identical207 original windows and37 gallery windows;83 windows calibrate thresholds and87 are TRAIN diagnostics. Full11 tap scores and thresholds are reused unchanged. The epoch3 encoder and fusion scaling have already seen these diagnostic observations; no independent validation claim applies.

| Method | TRAIN EER | Actual FAR at1% calibrated target | FRR |
| --- | ---: | ---: | ---: |
| tap11 | 41.00% | 0.00% | 98.85% |
| key | 50.96% | 1.15% | 98.85% |
| imu | 44.83% | 0.38% | 94.25% |
| joint | 45.98% | 0.77% | 94.25% |
| key_imu_tap11 | 44.83% | 0.00% | 95.40% |
| joint_tap11 | 44.06% | 0.00% | 91.95% |

Every original diagnostic claim is available; base and shared-intersection metrics agree. Adding touch to the joint encoder modestly improves these TRAIN metrics, but all methods remain poor. No DEV-dependent choice, new DEV read, or test access occurred. See `tap_paired_train_v2/paired_train_complete.json`. The original v1 plan was preparation-only and superseded before scoring to enforce command-line path binding.

## Frozen TRAIN-selection transfer: no useful discrimination

The exact60 original selection windows retain15 enrollment and45 probe
windows. Personal galleries use only enrollment taps and embeddings; all
encoder weights, six-method thresholds and fusion scales remain frozen.
Both accounts enrolled, and all original claims were available.

| Method | Selection EER | Actual FAR at transferred1% target | FRR |
| --- | ---: | ---: | ---: |
| joint_tap11 | 51.11% | 15.56% | 91.11% |
| tap11 | 64.44% | 0.00% | 100.00% |
| key | 46.67% | 4.44% | 95.56% |
| imu | 33.33% | 6.67% | 91.11% |
| joint | 35.56% | 2.22% | 93.33% |
| key_imu_tap11 | 51.11% | 11.11% | 95.56% |

Only tap11 passes the zero-observed-false-accept gate, but it rejects all45
genuine probes. Status is **no_useful_discrimination**, not a useful winner.
No threshold was recalibrated to rescue the result. With45 impostor claims,
one false acceptance already means2.22% observed FAR; no population-level1%
guarantee follows from zero observations. The same selection windows already
chose the encoder checkpoint, so these are TRAIN-selection results.
See `tap_selection_v1/selection_complete.json`; the original failed selection
must accompany any later evaluation-only DEV comparison.

## DEV touch comparison: enrollment infeasible

The frozen evaluation-only follow-up extracted5,451 tap vectors across all68
existing windows. Every window meets the five-tap scan minimum, but participant
556357 has77 support taps, below the unchanged80-tap enrollment requirement.
The other accounts have2,344,80 and686 support taps. All four galleries are
required, so the runner stops before encoder inference and reports no new
FAR/FRR/EER. Empty metrics do not mean zero errors.

See `tap_dev_v1/dev_complete.json` and `dev_tap_extraction_results_v1.json`.
No account was dropped, no probe used for enrollment, and no threshold or
minimum was relaxed after DEV exposure. Both accounts without genuine probes
remain explicit. No API scoring release exists for this infeasible comparison;
the earlier encoder-only API evidence is a separate experiment.
