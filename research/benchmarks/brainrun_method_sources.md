# BrainRun touch authentication method audit

Reviewed 2026-09-20. Primary papers and author repositories only; no participant records, arrays, weights, or external data bundles downloaded. Existing BrainRun feasibility and HMOG/BehaveFormer notes were reviewed first.

**Finding:** no inspected candidate establishes a runnable, faithful BrainRun touch-only SOTA reproduction. SwipeFormer is an attributable architecture candidate, but its public release has concrete execution and preprocessing gaps. A bounded TRAIN schema audit remains the next useful step.

## Candidate evidence

| Candidate | Attribution and available method | Decision |
|---|---|---|
| BrainRun touch-trace SVM, QRS 2020 | [Author-uploaded paper](https://www.researchgate.net/publication/345311892_Continuous_Implicit_Authentication_through_Touch_Traces_Modelling), [DOI](https://doi.org/10.1109/QRS51102.2020.00026): touch-derived features, SVMs and confidence-based decision accumulation. | Direct dataset relevance and plausible CPU baseline. Title/code searches and the first author's public repository listing did not locate its implementation. This is a search outcome, not proof that code does not exist. Do not invent feature equations, confidence updates or splits. |
| SwipeFormer, ESWA 2024 | [Author-hosted paper](https://biometrics.eps.uam.es/fierrez/files/2024_ESWA_SwipeFormer_Delgado.pdf), [author repository](https://github.com/BiDAlab/SwipeFormer): temporal and frequency Transformer branches, Gaussian positional encoding, CNN aggregation and 64-dimensional embeddings; triplet learning, Adam 0.001, ten-epoch validation patience. | Best new attributable architecture lead found here. Evaluated on Frank/HuMIdb and a private collection, **not BrainRun**. Authors claim improvements under their comparison protocol; neither current universal SOTA nor local reproduction follows. |
| TouchSeqNet, 2025 | [Original preprint](https://arxiv.org/html/2504.17271v1): masked-autoencoder pretraining followed by a Siamese temporal convolution/attention model. Reports accuracy/F1/AUC on Ffinger, BioIdent and Touchalytics. | Newer claim, but no attributable implementation located in the inspected paper or targeted searches. No demonstrated BrainRun result or low-FAR operating point. Not presently a stronger reproducibility choice. |

The separately found [BrainRun navigation implementation](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data) models motion-sensor measurements. It cannot supply the missing IMU channels in this non-sensor acquisition. Already-audited BehaveFormer remains a separate option; this note does not reclassify its IMU-dependent configuration as touch-only.

## SwipeFormer release: concrete blockers

Inspected public tree `ffa7c9039ba25d7bfebbfb89f58b0d6c4b058c8e` through the [GitHub tree API](https://api.github.com/repos/BiDAlab/SwipeFormer/git/trees/ffa7c9039ba25d7bfebbfb89f58b0d6c4b058c8e?recursive=1). It contains only two Python files plus README, image and a weights-named file. The latter has **two bytes**, not a usable trained network. No dataset/preprocessing/training module is present in that complete tree.

[SwipeFormer.py](https://github.com/BiDAlab/SwipeFormer/blob/ffa7c9039ba25d7bfebbfb89f58b0d6c4b058c8e/Code/Frank_database/SwipeFormer.py) imports absent `SwipeDatasetTriplet`, immediately loads external serialized datasets, hardcodes convolution layers onto CUDA, and runs evaluation at module scope. It specifies length50, channels11, ten layers, fifty heads and double precision. Its optimizer declaration is not a training loop. The paper instead describes nine layers; its table and prose disagree on head count. Resolve configuration provenance before claiming parity.

The evaluation also normalizes each account's binary-SVM scores using that account's evaluation genuine/impostor extrema. This is unsuitable for our frozen operational thresholds. Another line indexes validation embeddings with indices selected from test labels; provenance is not established by that code. Reusing architecture must not silently import these evaluation conventions.

CPU execution is **unverified**. Device-neutral layer construction and generated forward/backward checks are needed first; no runtime or accuracy estimate is justified here. Removing CUDA placement would be a disclosed portability change, not evidence of completed reproduction.

## BrainRun preprocessing gate

The [dataset paper, Tables 1–2](https://doi.org/10.3390/DATA4020060) provides gesture-level start/stop times and point fields `moveX`, `moveY`, `x0`, `y0`, `dx`, `dy`, `vx`, `vy`; its documented points have no individual timestamps, pressure or contact area. Taps contain one point. Consequently, do not fabricate uniform point times, missing pressure/area/IMU, or eleven SwipeFormer channels. Index resampling, coordinate normalization and touch-only channel reduction would be explicit adaptations requiring TRAIN-only selection.

First audit permitted TRAIN records for actual field availability, finite values, point counts, gesture types, repeated sessions and unambiguous user/device/game interval linkage. Keep taps separate from unsupported swipe encodings, preserve missing-capture coverage, and treat app-restart sessions as such rather than separate days. Fit normalization and any representation on fitting identities only; reserve selection and calibration identities. Device IDs remain join keys. No game aggregate may be renamed stimulus-onset reaction time. Any subsequent BrainRun model is a new declared adaptation until reference preprocessing, training and evaluation are reproduced.

## QRS 2020: recovered method contract

The [author-uploaded primary paper](https://www.researchgate.net/publication/345311892_Continuous_Implicit_Authentication_through_Touch_Traces_Modelling), Table III and Sections IV–V, specifies eleven features: endpoint separation in each axis; fitted-line slope; MSE, MAE, median absolute error and R² against that line; mean acceleration in each axis; mean coordinate in each axis. Axis lengths/positions were tried both raw and device-dimension-normalized.

Separate models handle Mathisis and Focus. Taps and short pseudo-swipes are excluded (70 ms is an example); a later variant retains 3–10 points. Owner enrollment uses a 0.75 split, with optional LOF filtering. Multiple owner-only one-class SVMs vary ν/γ; votes or summed signed certainty determine decisions. With training maximum distance M, certainty is −1 below −|M|, +1 above |M|, otherwise d/M; nonnegative sums accept.

Confidence changes are −a·t, +b for accepted predictions, −c otherwise: experiments use a=0, b=5%, c=9% Focus/15% Mathisis. Initial60%/threshold35% are examples; password unlocking resets confidence. Stateful evaluation counts attacker swipes before locking rather than equating that count to FAR.

### Unresolved implementation choices

Only indexed author-paper full-text passages were retrievable; direct HTML/PDF acquisition failed. The recovered passages do **not** establish acceleration finite differences/time denominators, regression orientation/intercept, vertical-line handling, signed versus absolute endpoint differences, or degenerate R² behavior. They also leave kernel, complete ν/γ grid, LOF parameters, feature scaling beyond the four geometry fields, random seed, session separation, confidence upper cap, and zero-M handling unresolved here. This is a retrieval-limited reproducibility assessment, not a claim these details are absent from every version of the paper.

For BrainRun, acceleration is the critical missing recipe: the documented point schema supplies velocities but no individual timestamps. Neither a fixed 15–20 ms denominator nor total-duration interpolation is an author-supported substitute. A geometry-only model, or a declared velocity-summary extension, could avoid fabricated timing, but would be a new baseline rather than this eleven-feature reproduction. Similarly, CPU suitability is an inference from small tabular inputs; an unspecified SVM ensemble size prevents a defensible runtime estimate. Preserve our fixed TRAIN roles and additional-verification policy rather than copying the paper's locking or unspecified within-owner split.
