# All-fit TRAIN consistency audit

The frozen audit read 25 metadata-selected nonempty-keypress sessions from four fit identities, sessions 1–16 only, through `HmogReader(phase='all_fit_schema')`. No other cohort, dev-support or test measurements were accessed. Script, helper and guard hashes were recorded before reads. No models, feature vectors or clock corrections were fitted.

| Identity | Sessions | Complete key pairs |
| --- | ---: | ---: |
| 717868 | 6 | 3,944 |
| 526319 | 7 | 7,160 |
| 986737 | 5 | 4,943 |
| 539502 | 7 | 4,901 |

All 51,451 numeric key events and 189,726 `_im` touch events reference an existing Activity ID and lie within that activity's relative-time bounds. Activity subject/session metadata matches the selected member. Ordinary and temporary touch streams also pass membership/bounds checks. Observed orientation values are 0 and 1. Under the proposed portrait-writing rule, 50,238 key events and 185,035 `_im` touch events qualify; one session contains 1,213 landscape key events and 4,691 landscape `_im` events. No orientation or identity was silently reassigned.

Exact key-anchor coverage in `_im` is 97.80–100% across sessions. In all 20 sessions with a temporary stream, its full 11-column numeric-row multiset exactly equals ordinary touch plus `_im`, including multiplicities. Five sessions lack the temporary file. This establishes numeric row equality, not byte-format equality. Concatenating all three streams would duplicate measurements.

## Proposed clock strategy and limits

Key/touch event times show direct correspondence; preserve key holds on their relative event clock. Recorded touch `SysTime` and sensor `SysTime` offer a common *logging* timeline without equating Android uptime and elapsed-realtime origins. Matched key/touch anchors can associate a key event with its touch logging timestamp, but this does not prove simultaneous physical sampling or remove sensor buffering, callback delay, clock adjustments, or batching.

For a later protocol, broad windows on recorded absolute time may be defensible as **logging-time co-occurrence**, provided TRAIN checks establish within-session monotonicity and report delayed/batched rows. Event-centered millisecond sensor responses require stronger synchronization evidence. Do not infer an offset correction from dev measurements. Do not use key logging order for hold times, map unmatched keys by nearest sequence position, or claim clock equivalence from similar offsets. This audit applies no retiming and extracts no modeling features.

Four synthetic tests pass, covering key-pair ordering/ambiguities and independent activity/time/orientation eligibility. Full aggregates, preregistration and executed source are `fit_consistency_results.json`, `fit_consistency_plan.json`, and `fit_consistency_source.py`. The four-person, archive-size-selected TRAIN evidence does not establish population coverage or authentication performance.
