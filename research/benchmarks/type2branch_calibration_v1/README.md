# Actual inner-TRAIN calibration

Selected trained epoch2checkpoint restored successfully with optimizer iteration300. Five gallery andtenprobe windows for each16calibration identity, no fitting/selection updates. Frozen global threshold-1.4094097841959736:24/2400false accepts(1%),121/160false rejections(75.625%). Pooled discreteEER20.7083%;macroEER15.6667%. This is poor calibration utility, not a DEV result or an improvement claim.

Scores and complete-training checkpoint shard hashes independently verified. Threshold is frozen for subsequentDEV; no adjustment based onDEV is permitted. Source-order/Average/residual and small-cohort protocol adaptations remain. Initial3epochbudget does not establish convergence or source-curriculum training.
