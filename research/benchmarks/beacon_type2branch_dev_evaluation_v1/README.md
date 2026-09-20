# Frozen paired DEV results

All three identities and243 claims (81 genuine,162 impostor). No DEV-based selection or threshold adjustment.

| Model | FAR | FRR | EER |
|---|---:|---:|---:|
| typenet | 0.00% | 100.00% | 35.80% |
| sapimouse | 66.67% | 56.79% | 59.26% |
| handcrafted_behavior | 2.47% | 97.53% | 42.90% |
| old_dual_neural | 54.94% | 59.26% | 59.26% |
| old_hybrid | 45.68% | 30.86% | 38.27% |
| type2branch | 11.73% | 80.25% | 43.21% |
| type2branch_pointer | 64.81% | 59.26% | 59.26% |
| type2branch_hybrid | 24.69% | 67.90% | 40.74% |

FAR/FRR use thresholds frozen for <=1% FAR on TRAIN calibration. That target fails to transfer for most models. EER is a DEV diagnostic, not a deployed threshold. Type2Branch hybrid trades fewer false accepts for more false rejects than old hybrid and has worse EER. Pointer fusion does not improve EER. No overall improvement claim.

140 enrollment/probe windows,81 below25 keys. Three independent people, gameplay domain, previously observed DEV cohort and source-transfer adaptation sharply limit generalization. Actual encoder preprocessing was offline; API fusion replay completed with1,944 exact scores and zero decision flips; see ../beacon_type2branch_api_replay_v1/. All pooled and per-probe metrics independently recomputed; prior34columns and claim order verified.
