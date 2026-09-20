# Paired TRAIN fusion

All24 candidates fitted on three TRAIN identities, selected on two other TRAIN identities, and calibrated on two reserved TRAIN identities. Eight models remain for matched DEV comparison; no DEV-based model selection.

| Model | C | Selection FRR at <=1% FAR | Calibration FRR at <=1% FAR | Calibration EER |
|---|---:|---:|---:|---:|
| typenet | 0.01 | 100.00% | 100.00% | 51.39% |
| sapimouse | 0.01 | 100.00% | 4.17% | 4.17% |
| handcrafted_behavior | 1.0 | 98.04% | 68.06% | 2.78% |
| old_dual_neural | 0.01 | 98.04% | 15.28% | 8.33% |
| old_hybrid | 0.01 | 96.08% | 2.78% | 2.78% |
| type2branch | 0.01 | 88.24% | 91.67% | 45.83% |
| type2branch_pointer | 0.01 | 98.04% | 6.94% | 5.56% |
| type2branch_hybrid | 0.1 | 96.08% | 9.72% | 8.33% |

Calibration includes72 genuine and72 impostor comparisons. The1% constraint permits zero false accepts at this sample size; it does not establish population FAR below1%. Strong selection/calibration variation across very few identities limits conclusions. Saved candidate choices, thresholds and calibration scores independently reproduced. DEV validation pending.
