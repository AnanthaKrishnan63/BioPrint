# BEACON Type2Branch transfer feasibility audit

Read only the original fitting identities P002/P003/P006 and their six full TRAIN keyboard recordings. Split/role ledgers and source/input hashes were verified before observation parsing. Existing dual-neural paired window starts were reused from the frozen TRAIN audit; pointer measurements were not reread and eligibility was not recomputed.

All observed elapsed timestamps and durations convert exactly to integer milliseconds; all key labels map successfully. The existing140paired windows include30with fewer than25real key events. Stable onset sorting is an explicit transfer adaptation, and auto-repeat overwrites in the source recorder cannot be recovered. Holds derive from elapsed release minus elapsed press; independent Duration measurements are audited rather than substituted.

Three generated tests passed. The audit creates no embeddings, fitted models, thresholds or recognition metrics, and reads no selection/calibration/DEV/test observations. It does not select an exclusion policy. Applying the KeyRecs25-event minimum would change the paired window set and gallery composition; any next experiment must explicitly preserve/account for coverage and compare methods on the same actual BEACON participant/session/window records. No artificial cross-dataset identity joins are permitted.
