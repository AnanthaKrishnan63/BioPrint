---
title: Keystroke dataset sources and modern baseline rationale
tags: [research, datasets, keystrokes, benchmark]
updated: 2026-09-20
---

# Keystroke sources and baseline rationale

## Acquired dataset

KeyRecs (2023) provides fixed-password repetitions and variable-text transcription from 99 participants. It includes two acquisition sessions, but these should not be represented as separate days. The authors call the transcription track free text; it is not spontaneous composition. [Dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10474054/).

The [versioned Zenodo source](https://zenodo.org/records/7886743) declares CC-BY-4.0 in its API metadata. Cite Dias, Vitorino, Maia, Sousa, and Praça and the dataset DOI `10.5281/zenodo.7886743`. Downloaded originals are unchanged and their published MD5 checksums verified. Demographics were not needed or downloaded.

| File | Bytes | Published MD5 |
|---|---:|---|
| fixed-text.csv | 5,676,298 | f3c2ef3a42625f4df7183ea36d4543db |
| free-text.csv | 25,402,938 | a5ca6fcb0970cfdcd8eb958b3fe9f22a |

Both together occupy 31,079,236 bytes, well below the per-dataset 500 MB limit. Source metadata and SHA-256 hashes live in `data/keyrecs/`. Files should remain local research data rather than being served by the application.

## Frozen partition and access controls

`scripts/data_keystrokes.py` reads participant/session metadata before any timing parsing. A stable SHA-256 ordering of participant IDs reserves 20 identities entirely for sealed test use. The remaining 79 identities use session 1 for training and session 2 for development validation. Both tracks share the same identity partition. No test measurements are parsed by the adapter, and requesting a test split raises an exception.

| Track | Training rows | Dev rows | Sealed rows, metadata count only |
|---|---:|---:|---:|
| Fixed repetitions | 7,878 | 7,896 | 3,998 |
| Free digraphs | 223,078 | 228,133 | 111,385 |

Training is split chronologically into 50% fitting, 25% internal model selection, and 25% operating-threshold calibration. Development is loaded only after models, transforms, and thresholds are frozen to disk. This within-session calibration is weaker than calibration on separate days and is disclosed. A future explicitly authorized test evaluation would need its own enrollment protocol for the reserved identities; this task does not open test data.

The existing CMU evaluator has previously used its historical test sessions. Consequently CMU is a legacy development benchmark, not a pristine holdout. Do not invoke the legacy whole-data loader when enforcing the new split policy.

## Modern methods and feasible implementations

**Type2Branch (TIFS 2025)** is a strong modern sequence-model reference. It combines recurrent and convolutional branches, attention, population-relative timing features, Set2set loss, and a curriculum. Its reported desktop mean per-subject EER is 0.77%, while global-threshold EER is 3.25%. Those are different evaluations. Training uses large KVC populations and is not reproduced here; the authors' code is GPL-3.0. [Paper](https://arxiv.org/abs/2405.01088), [official implementation](https://github.com/lsia/tifs-type2branch).

The subsequent bounded feasibility audit is recorded in [[type2branch_feasibility]]. It identifies missing compatible reference dependencies and external-feature generation, exact architecture/loss contracts, and the mismatch between the author identity split and available KeyRecs population. Type2Branch remains a researched reference; no implementation or reproduced performance is claimed.

**TypeFormer** is a mobile sequence-transformer reference, not an appropriate numerical comparator for desktop password repetitions. Its documented mobile raw archive is 5.6 GB compressed, outside the requested limits; no such archive was downloaded. [Official code and acquisition instructions](https://github.com/BiDAlab/TypeFormer).

**Zhang et al., Electronics 2026, 15(11):2325** provide a hardware-relevant KeyRecs baseline: per-user one-vs-rest logistic regression using train-fitted standardization, 47 fixed-text timing features, and windowed free-text timing summaries. Their free-text recipe uses windows of 30 digraphs with stride 10. [Primary paper](https://www.mdpi.com/2079-9292/15/11/2325).

Our CPU benchmark implements that **method family**, not an exact reproduction: positional fixed columns preserve duplicate names; free windows contain 50 nonoverlapping digraphs summarized by seven statistics over five timing channels. Train-only median imputation, quantile clipping, and standardization precede LR. Shrinkage LDA and 128-tree ExtraTrees provide compact alternatives. Hyperparameters are selected only inside training; no numerical SOTA claim follows from this adaptation.

## Execution and interpretation

```bash
~/miniconda3/envs/bigidea/bin/python scripts/data_keystrokes.py --download
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  ~/miniconda3/envs/bigidea/bin/python scripts/keystroke_benchmark.py --track both
```

The script uses repository-local `.research-deps` for scientific dependencies and writes frozen configurations, fitted models, dev scores, and metrics under `results/keyrecs-v1/`. Report EER and diagnostic oracle FRR@FAR separately from achieved FRR/FAR at the frozen train-calibration threshold. User-bootstrap intervals express user heterogeneity; repeated impostor comparisons are not independent security trials. Accuracy on these tasks cannot establish live browser spoof resistance, device independence, or generalization to unknown attackers.

Related: [[../big_idea/13 Roadmap]], [[../big_idea/10 Basics and Glossary]].

## Authentic checkpoint availability audit

The official Type2Branch repository tree at commit `6ddd9b0a13a0baf3f5631c73d26e2c2f67b08a1b` contains 17 source/documentation files and no checkpoints; the releases API returns an empty list. The official HOWTO explains training from supplied KVC raw data and provides no pretrained download. API snapshots are saved in `references/type2branch/`; inspected source is accompanied by its GPL license. This establishes absence from the inspected official distribution, not proof that no checkpoint exists anywhere. [Official training instructions](https://raw.githubusercontent.com/lsia/tifs-type2branch/main/HOWTO.txt).

The official TypeNet benchmark distributes precomputed embeddings only after a signed license agreement and email request; no request was sent. Those embeddings would not supply a transferable raw-sequence model. [Official benchmark](https://github.com/BiDAlab/TypeNet).

## TypeNet architecture transfer experiment

`scripts/typenet_benchmark.py` implements the paper's 200,458-parameter two-layer 128-unit LSTM with input/between-layer normalization, 0.5 inter-layer dropout, 0.2 recurrent dropout, and squared triplet loss with margin 1.5. It compares mean Euclidean distances to five enrollment embeddings. [Primary architecture and training specification, sections IV-A–D](https://arxiv.org/html/2101.05570).

The bounded CPU experiment is explicitly a **small-data architecture transfer**, not reproduced Aalto/KVC SOTA. KeyRecs supplies text key labels rather than original numeric event keycodes: mapped ASCII-like values are normalized by 255 and unknown keys become 0. Four temporal channels retain seconds. Timing imputation/clipping is learned exclusively on fitting rows. Two predeclared learning rates include the source value 0.05 and an adaptation 0.001; each gets at most 20 epochs × 20 batches of 32 triplets, with training-only selection/early stopping. The test identities remain sealed. No unpublished source weights or guessed pretrained model are represented as authentic.

## Isolated research website and API replay

The optional research app uses public benchmark artifacts and does not import the participant server or open its SQLite database. Start it **only when wanted**, bound to loopback:

```bash
PYTHONPATH=.research-deps:code/bioprint OPENBLAS_NUM_THREADS=2 \
  ~/miniconda3/envs/bigidea/bin/python -m uvicorn research_api:app \
  --host 127.0.0.1 --port 8011
```

Open `http://127.0.0.1:8011/` for the dashboard. Endpoints include:

- `GET /api/datasets` — ready datasets and readable split names.
- `GET /api/datasets/keyrecs-fixed/train/raw?offset=0&limit=128` — paginated source timing rows.
- `GET /api/datasets/keyrecs-free/dev/samples` — model-ready development samples.
- `POST /api/models/keyrecs-fixed/score` with `{"features":[[...]],"model":"train_selected"}` — frozen scores, thresholds, decisions.
- `GET /api/results/cmu` — frozen CMU development report, with historical-exposure warning.

All `test` paths are refused before data access. Client IP, Host and Origin must be loopback/local; no CORS permission is granted to remote websites. The service should never be attached to the LAN reverse proxy. Stop it with Ctrl+C.

`scripts/api_dataset_replay.py --in-process` tests the real ASGI routes without starting a server. With an explicitly running server, omit that flag and use `--base-url http://127.0.0.1:8011`. The replay reads development features through GET, scores through POST, compares with offline artifacts, and reports frozen-threshold rates. Use a new `--output` path to preserve earlier reports; `--datasets keyrecs-typenet` can verify only a newly trained model. The TypeNet entry appears only after training and frozen dev validation finish.

## Paired BEACON representation transfer

`scripts/beacon_neural_benchmark.py` transfers the frozen KeyRecs TypeNet encoder into the original paired30second BEACON gameplay windows. The S1 personal enrollment ledger explicitly marks support as training; global fusion uses only training identities. Pynput key labels map explicitly to ASCII-like codes; elapsed seconds are checked against recorder durations. Short sequences use their last valid state. Auto-repeat logger semantics and gameplay-to-transcription shift remain limitations.

`scripts/beacon_dual_neural_benchmark.py` adds the independently source-trained SapiMouse FCN described in [[pointer_sources]]. Its preregistered protocol requires129 raw pointer positions per accepted window, preserves all event types and order, and averages complete128-displacement block embeddings. No synthetic modality joins, padding or encoder retraining occurs. One train and one dev enrollment window are excluded; all identities and243 dev comparisons remain. Every V2 comparator uses the same eligible support/probes. The final source FCN SHA256 is pinned in `beacon-neural-v2/declaration.json`.

| V2 model | Dev diagnostic EER | Achieved FAR at training1% threshold | FRR |
|---|---:|---:|---:|
| TypeNet transfer |35.80%|0.00%|100.00%|
| Handcrafted behavior |42.90%|2.47%|97.53%|
| SapiMouse transfer |59.26%|66.67%|56.79%|
| Dual neural |59.26%|54.94%|59.26%|
| Hybrid behavior |43.21%|4.32%|92.59%|

These are negative transfer results on three dev identities, not a deployment improvement or a SOTA reproduction. The frozen1% operating point fails to transfer; TypeNet's low FAR is achieved by rejecting every genuine probe. No thresholds or models were changed after dev inspection. V1 artifacts remain separately preserved.

The isolated research API exposes `GET /api/neural-beacon/v2/pairs?split=dev`, `POST /api/neural-beacon/v2/score`, and `/results`. Pairs include33(V1) or34(V2) owner-relative features with uniform larger-is-impostor scores. Test requests are refused. Run `scripts/beacon_neural_api_replay.py --version v2` once to compare every endpoint score and frozen decision against saved offline models; no listening server is required.

## CMU account-specific baseline optimization

`cmu-account-selection/` preserves a preregistered follow-up using the original nine SVM C/gamma combinations. Each account selects its own combination on sessions2–3; exactly10 session1 enrollment samples, features, background data, and session4 calibration remain unchanged. No refitting follows selection. This is an adapted baseline optimization with prior global-dev and legacy-corpus exposure, not a SOTA reproduction or independent holdout claim.

One frozen sessions5–6 comparison gives EER14.32% versus global15.50%. At trainingFAR1% thresholds, achieved FAR/FRR are0.95%/67.43% versus1.00%/72.98%; at trainingFAR5% thresholds,4.87%/38.00% versus4.90%/44.76%. Paired-account EER-difference95%interval[−2.56,+0.26]percentagepoints includes zero. FRR-difference intervals favor account selection, but account resampling does not fully capture shared-impostor dependence. High remaining rejection rates preclude claiming the login problem solved. Full model choices, frozen thresholds, per-account metrics, bootstrap deltas and offline matrices are retained; no development-based retuning occurred.
