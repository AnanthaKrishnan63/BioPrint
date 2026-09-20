# Enrollment-only pointer score normalization

TRAIN-only follow-up on the frozen source FCN and existing five-block cosine
baseline. Z-normalization estimates per-account mean/std from673 decisions
across60 separate representation-TRAIN background identities. Enrollment comes
from12 calibration identities; their probe scores never fit normalization.

| Method | TRAIN EER | TRAIN FRR at1% target | Actual FAR |
| --- | ---: | ---: | ---: |
| Existing cosine | 14.16% | 48.57% | 0.779% |
| Z-normalized cosine | 12.21% | 54.29% | 0.779% |

The prespecified FRR-first selector retains cosine. Its three thresholds and
calibration metrics match the original frozen report exactly. This candidate
was not promoted; no new DEV or test observations were read. Earlier cosine
DEV/API results remain separate evidence, not a new validation of Z-norm.

This is a transfer hypothesis from [verification score-normalization research](https://www.isca-archive.org/interspeech_2018/shi18b_interspeech.pdf),
not a modern mouse SOTA reproduction. The small calibration population and
correlated claims cannot establish population low-FAR guarantees.

`plan.json` records the protocol, roles, checkpoint and executable hashes;
`train_complete.json` contains all operating points and the selected method.
The script has separate `prepare` and `run` actions, invoked through
`bash scripts/research.sh`. Existing outputs refuse overwrite.
