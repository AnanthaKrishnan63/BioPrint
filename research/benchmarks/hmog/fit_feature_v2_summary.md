# Strict v2 extraction: no eligible windows

The frozen strict protocol produced **zero eligible windows from 332 candidates** across all 25 authorized fit-TRAIN sessions. Training cannot proceed under this contract. No criteria were relaxed, and no selection, calibration or dev measurements were accessed.

Nonexclusive exclusions:

| Reason | Windows |
| --- | ---: |
| Fewer than 52 uniquely matched complete portrait pairs | 130 |
| Invalid/extra/missing intervening key events in retained 52-pair span | 202 |
| CANCEL/OUTSIDE/multitouch action | 222 |
| Touch pointer count other than one | 222 |
| Invalid native-time contact span | 136 |
| Nonportrait accelerometer / gyroscope | 10 / 10 |
| Nonportrait touch | 9 |
| Empty accelerometer / gyroscope bins | 1 / 1 |

The 130 insufficient-pair windows and 202 invalid-intervening-event windows partition all candidates. The latter rule intentionally prevents constructing digraph/trigraph features by silently bridging omitted key events. The earlier permissive feasibility count of 208 therefore does not support this stricter extraction contract.

The exclusive archive `datasets/hmog/fit_features_v2.npz` contains correctly shaped empty float32 arrays `(0,50,10)` and `(0,100,24)` plus empty typed subject/session/role/activity/window metadata. Its SHA256 is `912cdbac7fdafa95b14c4171a9b6871f8a2b0dd8d9de42a61c32e17c26e6f635`; compressed size is 1,334 bytes. Consumers must reject an empty training population.

Four generated-data extraction tests passed before reads: valid contract, cancellation/multitouch/rotation rejection, intervening invalid-key rejection, and sensor orientation/native-clock checks. Plan, source snapshot, dependency hashes and per-session quality counts are preserved in the `fit_feature_v2_*` artifacts. Native sensor order was checked without comparing sensor nanoseconds to the unrelated Activity/touch relative-clock origin. No tap features were fabricated, no models fitted, and no performance claims follow from this run.

Further work requires a separately frozen, explicitly adapted observation contract justified using TRAIN evidence only. Keep this negative feasibility result; do not relabel a more permissive replacement as v2.
