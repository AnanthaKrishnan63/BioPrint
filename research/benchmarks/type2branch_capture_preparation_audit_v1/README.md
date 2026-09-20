# Short-capture DEV preparation audit

The preparation script requires completed paired training and matching four-length calibration before accessing DEV payloads. It verifies frozen protocol, calibration/checkpoint/score hashes and population provenance, excludes sealed identities before timing parsing, prepares the exact full gallery cohort, and accounts for every probe including short captures requiring additional verification. Required-prefix errors fail the experiment rather than producing subset metrics.

79 generated gate/policy tests passed in 0.51 seconds. An actual premature invocation failed at the missing paired comparison report before creating DEV output. Real DEV preparation remains pending.

The control arm completed 600 total optimizer updates. Four saved TRAIN selection score matrices were independently checked against each checkpoint summary and checkpoint shard hashes receipted after completion. Calibration now requires those independent completed-arm receipts, including the mixed arm when it finishes. The initial runner audit snapshot is historical; the updated runner is snapshotted here. No DEV performance is claimed.
