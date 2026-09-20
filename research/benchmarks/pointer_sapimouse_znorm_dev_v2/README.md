# Frozen SapiMouse normalization comparison

| Method | DEV pooled EER | Actual FAR | FRR |
| --- | ---: | ---: | ---: |
| TRAIN-selected cosine | 13.87% | 1.66% | 43.42% |
| Unselected Z-normalized cosine | 11.73% | 0.74% | 60.53% |

Both use their frozen1% TRAIN FAR-target thresholds, the same encoder and
641 input events per decision. All24 DEV identities contribute76 genuine
and1748 impostor claims. The lower candidate EER/FAR costs higher rejection;
cosine remains selected under the original TRAIN rule. DEV was previously
exposed, and this comparison does not provide a fresh holdout.

Normalization statistics use only representation-TRAIN background recordings
and existing TRAIN-support centroids; probe values never fit them.
Original cosine scores and account/session metadata remain bit-identical.

The isolated `/api/pointer-znorm` research router exposes DEV samples, both
comparison scorers and results. Actual in-process ASGI replay reproduced3648
scores bit-for-bit and all three threshold decisions. Test, remote and
cross-origin requests were rejected. No listener was started or production
authentication policy changed.

Scripts: `pointer_sapimouse_znorm_dev.py` (prepare/run) and
`pointer_sapimouse_znorm_release.py` (export/replay), through
`bash scripts/research.sh`. Plans and outputs refuse overwrite.
The v1 preparation was superseded before measurements by v2's additional
record-role and frozen-threshold checks; both histories remain preserved.

Follow-up uncertainty: `../pointer_znorm_uncertainty_v1/` reports paired
probe-person bootstrap intervals conditional on fixed references/models.
The candidate-minus-cosine EER interval spans −5.47 to +3.98 percentage points;
the apparent EER gain is uncertain. No model selection or threshold changed.
