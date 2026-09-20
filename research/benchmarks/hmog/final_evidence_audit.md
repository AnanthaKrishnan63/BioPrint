# Final HMOG evidence audit

Audit scope: reports, source text, provenance references and opaque checkpoint hashing only. No participant archive or feature-array payload was decoded, no encoder ran, and no dev/API replay was repeated. Commands used `scripts/research.sh`.

## Scientific outcome

**The planned four-person DEV evaluation is infeasible.** All four accounts have support galleries, but participants 556357 and 737973 have no eligible dev probes. The remaining 31 probes come from only two people: seven from 219303 and 24 from 777078. They cover 31/154 inspected candidate windows (20.13%); 123 candidate windows yield no decision. Metadata-excluded empty/missing-keypress sessions retain unknown window counts, so 154 is not the complete recording-time denominator.

| Same-checkpoint branch | Pooled diagnostic EER | FAR at frozen 1% calibration target | FRR at that threshold |
| --- | ---: | ---: | ---: |
| Joint | 45.16% | 0/93 = 0% | 31/31 = 100% |
| Keyboard | 26.88% | 0/93 = 0% | 29/31 = 93.55% |
| IMU | 50.54% | 4/93 = 4.30% | 30/31 = 96.77% |

These are conditional rates over observed claims, not successful whole-cohort authentication. The joint zero FAR accompanies rejection of every genuine probe; it is not a low-FAR success. Including known missing genuine windows as not accepted gives joint 100%, keyboard 98.70% and IMU 99.35% genuine nonacceptance. Two participants have no genuine-score distribution; their missing FRR/EER must not be replaced by zero. Pooled EER is diagnostic interpolation, not an implemented threshold. The 93 impostor claims are dependent cross-account comparisons of 31 probes, not 93 independent attackers.

The joint model underperforms its same-checkpoint keyboard ablation on observed EER. Branch ablations are not separately optimized unimodal baselines. Neither this result nor exact API execution establishes SOTA or a complete BioPrint feature evaluation.

## Calibration and observation limits

Original identity-disjoint calibration was infeasible. The separately frozen fallback uses unoptimized enrollment observations from identities seen by the encoder, with 83 genuine and 249 impostor calibration claims. Its thresholds may be optimistic for unseen identities; the validation plan/report retains this qualification and the separate fallback-policy hash. Nominal target labels do not imply achieved DEV FAR or a population guarantee.

The smallest-archive cohort selection, retained-pair intervals across omitted events, numeric key-code effects, limited support (one window for each missing-probe participant), recording-time rather than hardware alignment, and 30-second observation cost constrain interpretation. No held-out score justified changing those rules.

## Integrity and execution evidence

Verified current hashes for base model protocol, completed selection, selected checkpoint, fallback calibration report/thresholds/policy, dev extraction report and its executed source snapshot. Validation source snapshots and current validator/selector/verification/training-core sources match frozen hashes. Extractor pure dependencies and current guard match the extraction plan. Selection-checkpoint, fallback-policy, upstream-report and feature-hash references agree across reports. Feature archive payloads were not reopened for this audit; their reported hash bindings were compared.

The saved API replay covers 31 probes × four galleries × three branches = **372 distances**, with zero acceptance flips at all frozen thresholds. Maximum absolute errors are approximately 1.08e-6 joint, 1.75e-6 keyboard and 3.22e-6 IMU; this is numerical agreement within the declared tolerance, not bitwise-identical scores. Replay source/release/report hashes match. Its `validation_status` remains `infeasible`, and coverage/missing participants are preserved.

## Presentation findings requiring explicit qualification

1. The frozen release manifest has `calibration_scope: null`. Exporter source reads a direct plan field, while the validator stores the scope inside `calibration_provenance`. The manifest's generic limitations omit seen-identity calibration optimism, although its fallback-policy hash is present and the full validation report is correct. Preserve frozen release/replay hashes and attach a clear metadata erratum or fix future exports; do not hide the caveat.
2. At audit time, the HMOG README said the API awaited a complete DEV release. The actual research release intentionally permits observed-probe replay with all four galleries and an infeasible cohort. Documentation should distinguish execution availability from scientific completeness.

No false SOTA claim was found in the reviewed HMOG README, validation report or release. The two presentation issues were reported to the root agent; this audit does not modify their files. The broader research objective remains incomplete.

## Addendum: presentation corrections

The root agent subsequently updated the README to describe partial observed-probe replay and its metrics explicitly. `metadata_erratum.md` documents the immutable release manifest's missing calibration scope. The future-export script now reads nested calibration provenance and preserves the optimism limitation; its current source therefore intentionally differs from the original frozen executed snapshot. The earlier hash agreement above describes the audit-time state, not this later source revision.

The root reports that `summary_v6` corrects the same three calibration-scope metadata fields while preserving 61 numerical results and 19 source hashes. Original release/replay artifacts, models and thresholds remain unchanged. This addendum records those reported corrections without new data reads or rerunning verification. The broader research objective remains incomplete.
