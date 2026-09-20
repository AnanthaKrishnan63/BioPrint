# Frozen raw-prefix TRAIN calibration

Selected mixed epoch4/500update encoder. Sixteen reserved TRAIN identities;80full gallery windows and160probes per length. Each exact-length inclusive threshold targets empirical1%FAR (24/2400impostor accepts). Thresholds were frozen before DEV scoring.

| Events | Threshold | FRR | Pooled discrete EER |
|---|---:|---:|---:|
| 25 | -7.270291457753 | 90.625% | 30.042% |
| 50 | -6.661237313948 | 88.750% | 20.562% |
| 75 | -6.324255031596 | 80.625% | 16.250% |
| 100 | -6.239031358101 | 77.500% | 16.250% |

These poor calibration results are not DEV performance. Raw-prefix synthesis and padding were applied before inference; training used feature-prefix augmentation, an explicit distribution difference. The prepared features and saved score hashes/counts were independently verified.
