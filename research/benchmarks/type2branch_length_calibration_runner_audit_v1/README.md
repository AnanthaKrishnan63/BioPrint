# Paired length calibration runner audit

The runner requires both completed 600-update arms, validates deterministic checkpoint selection and pair consistency, checks source/input hashes, and binds the selected checkpoint to its arm. It freezes checkpoint shard hashes before inference and checks saved optimizer iterations during restoration.

After completion, it will score the reserved TRAIN calibration features at 25/50/75/100 events with the selected encoder and unchanged chronological mean-Euclidean scoring. Each length gets a separate inclusive threshold targeting empirical FAR at most 1%, with FAR/FRR/discrete EER and dependent-comparison limitations reported.

59 generated gate/scoring/calibration tests passed in 0.43 seconds. An actual invocation rejected the currently missing paired comparison report before creating calibration output or reading calibration arrays. No inference or threshold fitting has occurred. Review corrected dependency overlay ordering before TensorFlow import. Actual model restoration and calibration remain pending.
