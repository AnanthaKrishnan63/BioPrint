# Completed paired TRAIN length study

Both arms started from the same epoch2/300-update model and optimizer and ran300additional updates each. Full input control took1315.14seconds and mixed25/50/75/100feature-prefix augmentation took1324.32seconds; peak memory stayed near2GB. All final unaugmented batch-stream hashes match and initial selection scores agree exactly.

The frozen rule minimizes mean TRAIN-selection oracle FRR at1% FAR across four lengths, then mean EER, then earlier updates, with exact arm ties favoring control. It selected mixed epoch4/500updates: meanFRR90.46875%,meanEER20.01042%. Control's selected epoch5/600updates had meanFRR91.875%,meanEER28.33333%. These remain poor TRAIN-only diagnostics, not deployable thresholds or DEV performance. The later mixed checkpoint has lower EER but worse primary FRR and was correctly not selected.

All four checkpoints per arm were retained; saved selection-score summaries were independently recomputed and checkpoint shard hashes receipted after completion. Raw-prefix calibration differs from feature-prefix augmentation because synthesis fallback order can change. Calibration, DEV validation and actual API replay are separate downstream steps. The failed original DEV protocol is unchanged.
