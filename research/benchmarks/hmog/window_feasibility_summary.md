# Frozen fit-TRAIN window feasibility, v1

The guarded audit used only the four fit identities' nonempty-keypress sessions 1–16 with `phase=fit_feature_feasibility`, `purpose=feature_feasibility`. Plan, script, contract, guard and pairing-helper hashes were frozen before reads. Three synthetic tests passed before measurement access. No features, models or recognition scores were calculated; no selection, calibration, dev, other-identity support or test data was read.

Of **332 full 30-second Activity-wallstart-anchored candidate windows**, **208 passed** and **124 failed** the frozen implementation. There were 123 windows below 52 complete uniquely matched portrait keypairs. One further window lacked at least one of the 100 half-open bins in both accelerometer and gyroscope; sensor exclusion counts therefore overlap. No counted windows failed the backward-anchor or sensor recording-clock checks.

| Fit identity | Enrollment candidates / eligible | TRAIN-fit candidates / eligible | Eligible TRAIN-fit sessions |
| --- | ---: | ---: | --- |
| 717868 | 30 / 10 | 42 / 22 | 9, 12, 13, 14 |
| 526319 | 77 / 46 | 33 / 30 | 10, 13, 15 |
| 986737 | 50 / 24 | 34 / 24 | 10, 16 |
| 539502 | 54 / 41 | 12 / 11 | 11, 12 |

Every fit identity has at least two eligible TRAIN-fit sessions and another eligible identity for negatives, so cross-session triplet sampling is possible under these checks. Enrollment windows remain excluded from global optimizer sampling. Availability is not evidence of accuracy or generalization.

## Exact v1 limitations

- Touch CANCEL and MOVE events are filtered out before contact pairing. Consequently a CANCEL between matched down/up events does not invalidate the pair in this implementation. The contract's cancellation policy remains unresolved; v1 does not validate cancellation-safe contacts.
- Portrait orientation is enforced for keys and matched touch endpoints. Sensor orientation is **not** checked. A window may contain nonportrait IMU observations, including activity intervals with orientation changes. The appropriate claim is portrait-key windows with same-activity, same-recording-interval sensor observations.
- Sensor Activity ID equality and absolute interval membership are checked; sensor-native timestamps are not tested against Activity relative bounds. Their clock origins cannot safely be equated without documentation. Previous key/touch Activity-bound checks do not establish sensor-native bound validity.
- Only complete 30-second windows ending by Activity wallend are candidates. Trailing fragments are omitted, and sessions without a nonempty keypress file were excluded prospectively by metadata. Candidate-window coverage is not total recording-time coverage.
- Backward matched anchors or a sensor SysTime backstep would reject windows under the frozen rule, even if buffering explains them. No retiming or relaxation was applied. Recorded absolute-time association does not establish hardware synchronization.

Any stricter cancellation/orientation protocol requires a separately frozen revision and fresh feasibility counts before training; preserve these v1 results. Full per-session counts and rejection diagnostics are in `window_feasibility_results.json`, with its plan and executed-source snapshot alongside it.
