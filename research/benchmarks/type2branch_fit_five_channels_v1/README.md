# type2branch_fit_five_channels_v1

Frozen TRAIN fitting sequences only: 47 identities, 705 sequences, 100 events per sequence. Original input arrays are preserved.

Five channels comprise unchanged normalized author base and two signed residuals. Valid residuals are observed normalized timing minus synthetic milliseconds/1000. Missing synthetic positions receive zero residual; validity is retained separately. Shape705x100x5, missing residual countsHT0/FT705, allfinite. This sentinel/clipping policy is an explicit adaptation, not a recovered paper recipe.

No selection/calibration/DEV/test observations were read. No encoder fit or recognition metrics yet. Source translation is not CLR runtime parity, and Average is not verified as the paper synthesis mode. See plan.json and report.json for hashes and limitations.
