# Type2Branch initial CPU training budget

Actual prepared KeyRecs features,47fitting and16inner-TRAIN selection identities. Three complete100-update epochs, batch10identities×15sequences, unchanged author architecture/Adam/Set2Set/generator. Source curriculum begins at epoch20 and is not reached by this initial budget.

Checkpoint selection is frozen to mean Set2Set loss on two fixed selection batches (sortedIDs0:10 and6:16; six IDs duplicated). Initialization and every completed epoch are saved with optimizer state. Python random state is retained, but exact resumed TensorFlow RNG trajectory is not verified. Calibration, DEV, and test arrays are never loaded by this script.

Status: running; inspect history.jsonl and eventual report.json. No recognition/convergence claim. All input/source hashes and explicit feature/runtime adaptations are in plan.json.

## Review correction

The two selection batches overlap FOUR identities (indices6–9), not six as incorrectly described above and in the frozen plan/source. Actual batch construction is unchanged. Their mean Set2Set loss is order-dependent and not an equally weighted biometric metric. Resume must restore generator cursor as well as RNG; after300updates cursor offset is18mod47. Initialization is eligible for selection and must be labeled untrained if it wins.

## Completed initial budget

Training process exited0 after300updates and1608.2254seconds (~26.8min), peakRSS2091.98MiB. Selectedepoch2loss0.3736602962 versus1.4901485443atinitialization. Allfourmodel/optimizercheckpoints retained; complete shard hashes are in checkpoint_receipts_complete.json. This initial3epochbudget does not reach epoch20curriculum or prove convergence. Calibration launched separately after checkpoint freeze.
