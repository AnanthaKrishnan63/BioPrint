# Compact nonlinear fusion search

This experiment did **not** find a better model on training selection. The original frozen logistic regression won for both the 32-dimensional behavioral representation and the 33-dimensional representation with advisory hardware context.

| Candidate | Behavior training-selection EER | Behavior + context training-selection EER |
|---|---:|---:|
| Original selected logistic regression | 41.18% | 41.18% |
| ExtraTrees, 128 trees, minimum leaf 3 | 49.02% | 47.06% |
| ExtraTrees, 128 trees, minimum leaf 10 | 52.94% | 49.02% |
| Random forest, 128 trees, depth 6, minimum leaf 3 | 49.02% | 47.06% |
| Histogram boosting, 100 iterations, 7 leaves, L2 1 | 49.02% | 49.02% |
| Histogram boosting, 100 iterations, 15 leaves, L2 1 | 50.98% | 50.98% |

The search was motivated by weak **training** selection performance in the earlier frozen linear/embedding experiments. Features, windows, original three fitting identities, two selection identities and two calibration identities were unchanged. Every new family was preregistered before fitting. All use balanced class weights and seed 20260920. Forest inference uses one job; boosting disables internal early-stopping splits. No candidate was refitted after selection, and no parameters were chosen from development results.

Training used 291 fitting, 102 selection and 144 calibration pairs, all from the existing paired recordings. Model selection used the original benchmark's EER calculation. The successful run took 33.2 seconds. An initial wrapper-packaging error occurred before frozen artifacts or dev access: a passthrough column selector needed its schema initialized on training features. The correction changed no classifier, feature or cohort. The failed attempt is documented separately.

Previously exposed development data cannot become a fresh independent validation set merely because a new method is introduced. This continued-development status is explicit in `preregistered.json`. The unused reserved test cohort remains sealed. With no training-selected improvement, no nonlinear model should replace the original reference or be described as a successful fusion improvement.

`frozen.json` preserves the first candidate scores and retained models. Before dev access, calibration was corrected in `frozen_v2.json` to match original BEACON's largest-observed-score threshold convention. The generic tabular helper's next-impostor threshold had selected a different point inside a training score gap; no weights or candidate choices changed. Exported selected/reference pipelines contain identical fitted LR parameters for each track; the column selector drops advisory context for the behavior-only track. Two generated-input tests verify cohort separation, source-protocol hashes and context exclusion without reading recordings.

## Wrapper verification on existing dev pairs

The existing frozen `beacon/dev_pairs.npz` supplied all 243 pairs; raw recordings were not reread for dev. Only the retained LR and identical reference were evaluated. The five rejected nonlinear candidates per track remain training-only results.

| Retained track | Actual dev FAR | Dev FRR | Original-reference decisions |
|---|---:|---:|---|
| Behavior | 2.469% (4/162) | 98.765% (80/81) | Every pair identical |
| Behavior + advisory context | 8.025% (13/162) | 86.420% (70/81) | Every pair identical |

These operating points use the original calibration 1% FAR target. Selected and reference wrappers match exactly. Strict ASGI replay passes all 243 pairs across four wrapper exports with zero probability discrepancy, exact returned thresholds and acceptance flags. Original reports and models were preserved.

**EER conventions differ between report adapters.** The new tabular report computes an interpolated ROC crossing; original BEACON reports the mean FAR/FRR at the nearest observed crossing. Thus the retained behavioral LR appears as 43.827% interpolated EER versus 44.136% under the original convention despite identical fitted classifier and operational decisions. This is a reporting convention, not an improvement. Within the new report, selected and reference EER are identical. The substantive result remains that training selected no better nonlinear model.

## Serving export integrity

`export_integrity.json` is an immutable export manifest created without overwriting existing artifacts. It hashes the schema, four serialized classifiers, corrected `frozen_v2.json`, preregistered protocol, executed script snapshot, original BEACON frozen configuration, original split manifest and feature script. Its own SHA256 is pinned in `research_modalities.py`:

`99514fc65662307bad9826e6dbf2c6030706ef5d7434579de5081e5f296f6ace`

For this dataset only, an uncached schema/model load verifies the manifest and all eleven referenced files. Missing or mismatching files produce HTTP 503 before joblib deserialization. The model is deserialized from the same verified byte buffer, avoiding a reopen between checking and loading. Existing cached objects are **not continually checked**; restart or explicitly clear caches when loading a deliberately reviewed new export. This is local export integrity, not a signed attestation of data collection.

Thirteen affected tests pass, including five tamper subcases for schema, model, corrected frozen configuration, preregistration and manifest. Each tamper case is rejected before deserialization even when a valid schema was already cached. Synthetic ASGI checks still return corrected thresholds. No measurements, fitted weights, thresholds or statistical reports changed during this integrity patch, and no dev replay was rerun.
