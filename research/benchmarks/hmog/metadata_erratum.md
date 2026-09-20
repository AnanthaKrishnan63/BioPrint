# Preserved release metadata correction

Independent review found that `release_v1/manifest.json` has a null
`calibration_scope`. The exporter looked at a direct plan field while the
validator stores the scope under `plan.calibration_provenance`. The pinned
`dev_report.json` already contains the full scope, policy hash and limitation:
calibration used disjoint enrollment sessions from encoder-trained identities
and may be optimistic for new DEV identities.

The frozen manifest, model, thresholds, validation and372-score API replay remain
unchanged. This erratum does not claim that the original manifest contained the
missing text. Its generic limitations field also omitted this calibration caveat.
Future exports now read the nested scope and limitation correctly. The original
export/replay source snapshot is preserved alongside its artifacts.

The same direct-field lookup affected the three HMOG rows' `calibration_scope`
in `summary_v5.json`; its top-level constraints already described seen-identity
calibration. `summary_v6.json` corrects only that scope metadata. All61 metric rows
retain their numeric results, and all19 source-report hashes remain unchanged.

Full-cohort DEV validation remains **infeasible**. This metadata correction does
not alter any score, threshold, decision, coverage count or scientific conclusion.
