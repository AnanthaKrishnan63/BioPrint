# Fit v3 retained-pair feature extraction

Frozen `model_protocol_v2.json` explicitly allows source interval formulas over successive **retained complete observed pairs**, including gaps caused by omitted events. These intervals are not a reconstruction of consecutive physical keystrokes. Strict v2's zero-window result remains preserved.

Extraction produced **207 eligible windows from 332 candidates**: 120 enrollment windows and 87 TRAIN-fit optimizer windows. Both core arrays are finite float32: keyboard `(207,50,10)`, IMU `(207,100,24)`. Subject/session/role/activity/window metadata is included. The private archive is `datasets/hmog/fit_features_v3.npz`, 2,035,861 bytes, SHA256 `2afe1a18164334e407beaf01dcaadc7613584fbc85d44fcdf29d0a8854cbdb36`.

| Identity | Enrollment windows | TRAIN-fit windows | Eligible TRAIN-fit sessions |
| --- | ---: | ---: | --- |
| 717868 | 10 | 22 | 9, 12, 13, 14 |
| 526319 | 45 | 30 | 10, 13, 15 |
| 986737 | 24 | 24 | 10, 16 |
| 539502 | 41 | 11 | 11, 12 |

Nonexclusive rejection counts: 124 windows below 52 unique matched pairs; ten windows with nonportrait accelerometer and ten with nonportrait gyroscope records; one window missing bins in both sensors. Each fit identity retains two or more eligible optimizer sessions. Enrollment must not train global weights.

Retained endpoints account for a median 81.25% of raw events in the retained span, range 66.24–97.20%. Median extra events are 24 per window, including 22 orphan/ambiguous events. Across 10,557 successive retained-pair intervals, the median duration is 302 ms, 99th percentile 2,245.04 ms and maximum 10,355 ms. Extra gap-event count has median zero and maximum 24. No gap threshold was optimized or imposed after observing these values.

The per-pointer parser retains actual action 0/5 DOWN and 1/6 UP events. CANCEL invalidates all active contacts in the activity; nonportrait, duplicate, repeated-down or invalid-count events invalidate affected contacts. Positive integral pointer counts above one are allowed. Unique exact key/touch anchors and complete same-pointer contacts are still required. Unrelated contacts no longer cause blanket window rejection. Sensor orientation, native-clock order, recording-clock order and complete half-open bins remain required; no native cross-clock equivalence is assumed.

Ten generated-data parser/extraction tests passed before reads. Plans, source snapshots, dependency hashes and archive binding are preserved in `fit_feature_v3_*`. Only authorized fit-TRAIN records were read. No models, tap-reference features, selection/calibration/dev/test access, learned normalization or latency correction occurred.
