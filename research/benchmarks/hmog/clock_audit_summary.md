# First-fit TRAIN clock/contact audit

Only identity 717868, sessions 3, 4, 9, 12, 13 and 14 were read through the role guard. Metadata selected every nonempty keypress TRAIN session; no other identity, selection, calibration, dev or sealed-test contents were read. Plans and script hashes were frozen before reads. No model was fitted, no raw rows exported, and no archive extracted.

## Findings

- Recovered 3,944 complete keypress pairs, with 544–884 per session. File-order event timestamps move backward 386–636 times per session. Ordering by event time is necessary; logging order is unsuitable for pairing. Missing-down events and ambiguous repeated-down cycles remain excluded rather than repaired.
- `TouchEvent_im.csv` matches 98.76–100% of unique key `(activity, event time, down/up)` anchors exactly. `TouchEvent.csv` matches none. `tempTouchEvent.csv` has the same key-anchor coverage as `_im`, and its row count equals the sum of the two touch streams in each audited session. Count equality alone does not prove exact full-row union.
- `_im` contains 16–54 duplicate contact anchors per session. Duplicate/ambiguous contact cycles are rejected. Neither source streams nor the temporary stream should be concatenated indiscriminately.
- Keypress and touch rows satisfy the expected numeric column widths and known action codes in this audited subset. Their correspondence supports shared key/touch event-time use for these sessions; it does not document `_im` semantics or establish coverage for other people.
- Sensor `SysTime − SensorTime/1e6` median offsets are 4.85–6.40 ms below corresponding `_im` `SysTime − eventTime` medians. Sensor offset standard deviations are 6.69–7.79 ms. These are observed logging-offset statistics, not synchronization corrections. Android touch uptime and sensor elapsed-realtime clocks have different sleep semantics; no dev retiming or equality assumption is justified.

## Corrections and artifacts

The first audit incorrectly expected six sensor columns and rejected those rows structurally. Existing authorized TRAIN probe metadata establishes seven sensor columns. Version 2 froze this correction before rereading the same permitted sessions, with zero invalid sensor rows. Original plan, results and source are preserved as `clock_audit_plan.json`, `clock_audit_results.json`, and `clock_audit_source_v1.py`. Use the corresponding `_v2` files for complete findings.

Three generated-data tests pass: event-time pairing despite reversed logging order, duplicate/missing cycles dropped, and repeated-down cycles rejected without fabricated releases. Full numeric aggregates are in `clock_audit_results_v2.json`; interpretation of source clocks is documented separately in `../hmog_schema_sources.md`.

This is a one-person TRAIN integrity audit, not biometric performance evidence or approval to unseal additional measurements.
