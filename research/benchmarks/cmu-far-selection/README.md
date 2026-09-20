# CMU selection for low false acceptance

This exploratory baseline experiment selects each account's SVM parameters for low false rejection at 1% selection FAR. It addresses an objective mismatch with EER selection; it is not a SOTA reproduction. Earlier CMU development results and historical full-corpus exploration were already known.

The preregistered grid is unchanged: C and gamma multiplier each in `{0.1, 1, 10}`. Each model uses 10 session-1 enrollment samples and the original 28 timing features/background. Sessions 2–3 select the lowest FRR at empirical 1% FAR, with ties resolved by EER then fixed grid order. Session 4 alone supplies final operational thresholds; selection thresholds are never deployed. Models are not refit. Sessions 5–6 provide the one frozen development comparison; sessions 7–8 remain sealed.

Training selection FRR improved from 51.16% for account-specific EER selection to 45.10%, while selection EER worsened from 9.21% to 11.77%. Choices changed for 28 of 51 accounts. These selection-set gains are optimistic. Training took 25.29 seconds on two CPU threads. All three prior frozen controls remain unchanged.

| Frozen model | Dev EER | Achieved FAR / FRR at training 1% FAR threshold | Achieved FAR / FRR at training 5% FAR threshold |
|---|---:|---:|---:|
| Original baseline |22.04%|1.10% / 74.20%|4.98% / 55.65%|
| Global SVM |15.50%|1.00% / 72.98%|4.90% / 44.76%|
| Account EER selection |14.32%|0.95% / 67.43%|4.87% / 38.00%|
| Account low-FAR selection |18.50%|0.99% / 64.10%|4.93% / 39.92%|

Against account EER selection, the low-FAR selector reduces FRR at the training 1% operating point by 3.33 percentage points (paired-account bootstrap 95% interval: −6.96 to −0.37). Achieved FAR changes by +0.037 points (−0.030 to +0.099). EER worsens by 4.17 points (+1.43 to +7.56), and FRR at the training 5% operating point changes by +1.92 points (−1.22 to +5.24). This is an operating-point tradeoff, not a universal improvement or production promotion.

Intervals use 2,000 paired account resamples, seed 20260920. They do not fully represent shared-impostor dependence, and no multiplicity adjustment covers the exploratory sequence of experiments. Rejection rates remain high.

`preregistered.json`, `frozen_models.json`, and `validation_plan.json` pin decisions and source/artifact hashes. `dev_results.json` contains account metrics and paired deltas; `dev_scores.npz` preserves four complete distance matrices with labels and subject order. Scores there are larger-is-impostor; the research API negates them. Training and validation source snapshots are retained. No post-development tuning occurred.
