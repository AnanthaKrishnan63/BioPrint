# HMOG paired-modality protocol proposal

Status: **proposal only, 2026-09-20**. This document does not authorize unsealing. Inspected official documentation, acquisition preregistration and receipt filename/size metadata only; no CSV member contents, features or outcomes were read. Freeze a machine-readable record-role manifest and parsing plan before opening TRAIN measurements.

## Sources and applicability

The [official dataset documentation](https://hmog-dataset.github.io/hmog/) describes randomized reading, writing and map tasks under sitting/walking conditions. Its [publisher repository](https://github.com/hmog-dataset/hmog) supplies the README rather than a complete CSV schema. The [authors' dataset poster](https://gzhou.pages.wm.edu/wp-content/blogs.dir/5736/files/sites/13/2018/09/SenSys14.pdf) describes touch and virtual-keyboard recordings. The [HMOG paper, section III](https://arxiv.org/html/1501.01199#S3) reports multiple collection days and possible device changes between visits. Thus session numbers do not establish elapsed-day chronology, task balance, or a persistent physical device. This is a smartphone typing/touch pilot, not desktop mouse validation or a reproduction of published HMOG performance.

## Proposed immutable roles

Retain acquisition identity assignments exactly:

| Cohort | Identities | Sessions 1–8 | Sessions 9–16 |
| --- | --- | --- | --- |
| Fit | 717868, 526319, 986737, 539502 | TRAIN enrollment | TRAIN global model fitting |
| Selection | 180679, 622852 | TRAIN enrollment | TRAIN candidate selection |
| Calibration | 962159, 663153 | TRAIN enrollment | TRAIN threshold calibration only |
| Dev | 556357, 219303, 777078, 737973 | TRAIN support only | DEV reporting only |

Parse numeric session suffixes, never lexicographic order. Sessions 17 onward are sealed TEST for every acquired identity, including fit identities. Four preregistered held-out identities and 84 unused identities remain unfetched. Unknown filenames or session numbers receive no read permission. Dev-cohort enrollment must not train the global representation, scaling, feature selection or fusion.

Receipt metadata shows nonzero `KeyPressEvent.csv` sizes in both proposed ranges for all 12 identities; this is not evidence of valid events or clock overlap. Selection identity 622852 has only one nonempty support keypress file. Some subjects have fewer than 24 directories. Do not move sessions, replace identities, or drop dev participants because support/probe data proves sparse. Report enrollment failures, missing modalities and eligible-window coverage for all four dev identities; unsupported accounts yield no verdict. Authentication error rates conditional on scorable windows must accompany coverage, not conceal failures.

## Clock, schema and pairing gates

Exact CSV column indices, timestamp units/epoch, event-time versus logging-time semantics, key down/up codes, pointer IDs, coordinate units/orientation, pressure scaling, and device identifiers remain unresolved. Metadata contains `TouchEvent.csv`, `TouchEvent_im.csv`, `tempTouchEvent.csv` and derived gesture streams. Their names alone do not establish which captures keyboard touches or whether streams duplicate events.

After role freeze, inspect only the first fit identity's allowed TRAIN headers/documentation and a bounded sample to resolve schema; expand to other fit TRAIN sessions for consistency checks. Record evidence and parser assertions. Prefer documented shared event time. Never align streams by row number, sequence shape, independent time-zero normalization, or dev-optimized offsets. If clocks differ, require documented conversion or explicit synchronization anchors; otherwise stop paired analysis. Validate monotonicity, resets, duplicate events, units and actual cross-stream overlap on TRAIN. Never concatenate sessions or bridge clock resets.

Propose nonoverlapping 30-second half-open windows on one shared session clock, anchored at the session's first valid timestamp across the selected streams. Both modalities must come from the same participant, session and interval. Require ten complete key-down/up pairs and ten complete touch contacts within each window; boundary-spanning events are excluded. These are proposed engineering rules, not published HMOG defaults. Freeze the exact touch stream, missing-event rules and clock conversion after TRAIN schema resolution and before any selection/calibration/dev evaluation. If the requirements prove infeasible on TRAIN, record a revised proposal before dev access; never relax them based on dev coverage or scores. Do not fabricate key releases, touch positions, modality pairings or device classes.

## Modeling and reporting gates

Fit population transforms and fusion only on fit identities; account profiles use their assigned enrollment. Select candidates only on selection identities. Derive the final operating threshold solely from calibration identities, without refitting after selection. Freeze features, candidate grid, seeds and observation budget before selection measurements are scored. Report unimodal and fused results on identical eligible paired windows; include actual time/event costs. Use real same-window impostor probes against other eligible accounts, never artificial cross-person modality combinations. Report pooled and per-account FAR/FRR at frozen thresholds, diagnostic EER, account/session counts and missingness. Two calibration identities and four dev identities cannot establish reliable low-FAR population guarantees.

## Budget, bias and publication

The acquisition selected the twelve smallest subject ZIPs, totaling 427,718,950 bytes, under a 500 MB cap. Recording length/completeness can correlate with identity, task and behavior: this convenience cohort is not representative of all 100 users. Retain that qualification in every result.

The official terms permit noncommercial educational/research use and prohibit redistribution without written permission. Keep participant archives and row-level derived biometric data private; publish code and aggregate reports, with source acknowledgment and the required disclaimer that the College of William and Mary is not responsible for this analysis. No production authentication deployment is justified by this proposal.
