# Strict development protocol

## Data boundaries

Every acquired dataset has three declared partitions: train, validation/dev, sealed test. Freeze the assignment using identity/session metadata before parsing observations. Never load, summarize, score, replay, or inspect test observations. Downloading an archive containing test partitions is permitted only where splitting requires it; loaders filter metadata first. Test identities/filenames are partition metadata, not feature analysis. Where feasible do not download reserved test files at all.

Use training-only internal fit, selection and calibration subsets. Choose hyperparameters and operating thresholds before loading dev. Dev validates a frozen model and does not select a winner or threshold. EER and ROC values on dev are retrospective descriptions; operational FAR/FRR use frozen training-calibration thresholds. Do not describe a dev EER threshold as a deployable threshold. Preserve failed experiments and source hashes.

CMU has historical exposure: old repository scripts ran and tuned on later sessions. Newly sealed sessions 7–8 are not claimed as a pristine test set. New partitions prevent additional access but cannot undo exposure.

## Comparisons

For each modality compare the existing approach (where transferable) to an evidence-backed baseline and a bounded improvement. Distinguish faithful SOTA reproduction, inspired CPU approximation, and older practical reference baseline. Published results with different users, enrollment size, split, sequence length or global/per-user thresholds are not directly comparable.

Report FAR (impostors accepted), FRR (genuine probes rejected), diagnostic EER, query count, user count, acquisition/session split, compute time, and uncertainty. Avoid independent-trial confidence claims for dependent all-pairs comparisons. New devices are a context axis, not evidence of a different person by themselves.

Joint optimization requires aligned recordings of the same actual participants. Never fuse unrelated CMU/Balabit/fingerprint identities, average their error rates, or fabricate cross-modality samples. API/synthetic integration checks may verify plumbing and attack scenarios but cannot substitute for real multimodal error rates.

## Resources and safety

Every source download is under 500 MB; total downloaded/extracted datasets under 5 GB. All new artifacts, dependencies, temporary data and caches are inside this repository. CPU only, normally two BLAS/OpenMP threads per worker. Do not access the live bioprint.db for experiments or overwrite production models/users. Replay public train/dev data only into an isolated database.

## Goal evidence

Completion needs: sources and reproducible downloads; guarded three-way splits; baseline and training improvements per available modality; frozen dev reports; scripts exercising dataset-derived requests through API; aligned multimodal validation or an explicit unresolved data requirement; master worklog and experiment log; runnable website and documented integration. Near-zero errors are an aspiration, not a required claim.
