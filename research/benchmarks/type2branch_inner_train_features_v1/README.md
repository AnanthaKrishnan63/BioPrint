# Inner-TRAIN feature preparation

Selection and calibration each contain16identity-disjoint users and240sequences (first15full100event windows/person). Both are disjoint from the47fitting users. Dataset session2DEV and sealed-test timing/key payloads were not decoded. The frozen fitting-only population model generated timing predictions without updates.

Each role starts a fresh deterministic fallback stream, traversing sorted users and chronological windows. Selection required29draws/channel; calibration28/channel. Each has240missing firstFT residuals, neutralized to zero with an audit mask. Original row order, approximate key codes, Average mode, and residual handling are explicit adaptations.

The scoring contract reserves first5windows as gallery and last10as probes. Selection may choose checkpoints; calibration may only set thresholds after model/score freeze. Calibration features are prepared but have not supplied fitting, checkpoint metrics, or thresholds. No recognition metrics exist for this candidate yet.

Eleven generated split-guard/adapter/residual tests passed. Shapes, identity membership, counts, finiteness, and output hashes were independently checked.
