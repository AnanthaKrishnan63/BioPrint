---
title: Pointer benchmark sources and protocol
tags: [research, pointer, benchmarks]
updated: 2026-09-20
---
# Pointer benchmark sources and protocol

## Summary

Balabit is the first reproducible pointer benchmark here, chosen for independently recorded sessions and an accessible publisher repository. This does not imply it is the newest or best possible dataset. Publicly available mouse research now includes learned temporal representations. Our immediate CPU comparator is a supervised tree ensemble, not a claimed reproduction of current SOTA.

| Primary source | Evidence and decision |
|---|---|
| [Balabit publisher](https://github.com/balabit/Mouse-Dynamics-Challenge) | Ten users; remote desktop interaction logs; official training sessions contain genuine activity. Official test files and labels are excluded entirely. Metadata shows 65 training sessions totaling 99,342,740 bytes. |
| [Antal et al. implementation](https://github.com/margitantal68/mouse_dynamics_balabit_chaoshen_dfl) | Research precedent for action features with per-user, two-class random forests. Our window features and session partition differ; published scores are not directly comparable. |
| [SapiMouse authors](https://github.com/margitantal68/sapimouse) | Newer controlled browser dataset, 120 participants, two sessions; fully convolutional feature learning plus one-class SVM. Strong next dataset for cross-user representation learning. Two sessions require participant-level train/dev/test partition and enrollment/probe separation inside each evaluation participant. |
| [DFL publisher](https://www.ms.sapientia.ro/~manyi/DFL.html) | Another open collection; suitable external replication candidate. Not yet downloaded. |
| [2025 LTMouseAuthen paper](https://arxiv.org/html/2504.21415v1) | Recent SOTA claim combines local ResNet and temporal GRU with velocity sequences; reported AUC does not establish superiority under our stricter session split. Faithful reproduction remains future work. |

## Frozen protocol

Download only official training session paths assigned to train or dev. Before reading measurements, hash session paths with a fixed salt; assign one session per user to dev, one to sealed test, one to training calibration, and the remainder to fitting. Filenames do not establish chronology, so this is **session-separated, not longitudinal**. Sealed test content is not downloaded. The official test is also never downloaded.

Extract nonoverlapping 128-movement-event windows, resetting at session boundaries, clock reversals, and gaps above five seconds. Use 29 kinematic descriptors without absolute screen coordinates, network timestamps, session IDs, or user IDs as features. Device/task confounding remains unresolved. A window is not equivalent to one login or one click.

Compare the existing production scorer's actual center/spread/deviation functions with per-user RandomForest and ExtraTrees models. Choose ensemble and leaf size by training-calibration FRR at 1% empirical FAR, breaking ties with calibration EER. Freeze selection and all thresholds **before** loading dev. Report dev FAR/FRR at training-selected 0.1%, 1%, and 5% FAR targets; actual dev FAR may differ. Dev EER is a descriptive curve statistic, never a deployed threshold or model-selection input.

The production pointer extractor needs submit-button geometry and keyboard-to-mouse context absent from Balabit. No artificial button geometry is invented. This validates a compatible feature/scorer pipeline, not the production target extractor. No identities are paired with unrelated keystroke datasets.

## Reproduce

```bash
/home/ananthakrishnan/miniconda3/envs/bigidea/bin/python scripts/pointer_download.py
PYTHONPATH=.research-deps:code/bioprint /home/ananthakrishnan/miniconda3/envs/bigidea/bin/python scripts/pointer_benchmark.py
```

The download script fetches publisher file metadata into `data/benchmarks/balabit/tree.json` if absent. Runtime verifies each accessed CSV against its publisher Git blob SHA. Results live in `research/benchmarks/pointer/`. Repeated evaluation requires explicit `--force`; inspect existing worklogs first.

Related: [[13 Roadmap]], [[00 Home]].

## SapiMouse architecture replication S1

Acquired publisher ZIP (8,056,327 bytes; CSV payload 33,433,632 bytes). Hash-ranked **identities**, before any measurement access, into 72 training / 24 dev / 24 sealed test. Training is further divided into 60 representation-learning identities and 12 unseen calibration identities. Sealed test archive members are never extracted or opened. No existing publisher train/test arrays are read.

The [authors' FCN source](https://github.com/margitantal68/sapimouse/blob/main/util/fcn.py) specifies 128/256/128 convolution filters, kernels 8/5/3, same padding, batch normalization, ReLU, global average pooling, and a softmax identity head. The PyTorch port preserves this architecture. Input is 128 successive absolute x/y displacement pairs, per-window z-scored over both channels. Settings preserve Adam, batch 16, 100 epochs, with the best checkpoint chosen using only representation-training identities' second sessions. Weight initialization and optimizer runtime are explicitly a framework port, not bit-identical reproduction.

For twelve training-calibration identities, enroll on their three-minute sessions and probe their one-minute sessions. Predeclare OCSVM default nu=.5/gamma=scale, enrollment-only sample-count score normalization, nu=.1, and one/three/five nonoverlapping-window aggregation. Select by training-calibration FRR at 1% FAR, then EER. Freeze choices before loading dev. Evaluate both the author-default OCSVM and selected adaptation, plus existing scaled-Manhattan on kinematic descriptors, at the same selected aggregation duration. Fit each dev identity's profile only from designated enrollment; no classifier learns from dev probes.

The published evaluation code includes score normalization using positive/negative evaluation scores and averaging that can cross impostor-user boundaries. Those operations are **not** reproduced: our score scaling uses enrollment data only, aggregation stays within each session, and thresholds are training-calibrated. These methodological differences prevent direct comparison with published EER. Author sources and exact Git blob checks are recorded in `pointer_author_source/provenance.json`; Apache-2.0 license retained.

```bash
python scripts/pointer_sapimouse_prepare.py
PYTHONPATH=.research-deps:code/bioprint OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/pointer_sapimouse_benchmark.py
```

Use the `bigidea` environment. Training exports its best checkpoint and history incrementally. `--resume-trained` continues calibration/evaluation from that checkpoint after an interrupted run; never use it to replace a live training process. Results refuse overwrite. This is an established deep baseline. The [2025 LTMouseAuthen feasibility audit](pointer_lt_mouse_audit.md) found missing architecture dimensions/training settings and no attributable implementation; faithful reproduction is unresolved, and no substitute model was presented as that result.

Archive metadata qualification: there are245 CSVs because three participants have extra recordings. Enrollment/fitting is defined by the `_3min.csv` suffix and heldout probing by `_1min.csv`, including all files with those suffixes; neither filenames nor the protocol prove elapsed-day chronology. Windows and score averaging never cross files. Claims of generalized authentication must also distinguish the roughly three-minute enrollment and multiwindow probes from the app's single reach-and-click signal.

Record-role clarification: “dev cohort” means identities unseen by the global encoder. Their three-minute enrollment/support files are explicitly **training records**, dedicated to account-profile fitting and excluded from global representation training. Only their one-minute probe files are **dev records**. This designation was frozen before dev-cohort analysis. Code asserts split/suffix consistency; dev probes never fit or normalize models. Earlier cohort-label metadata is retained solely as an audit record.

## Optional application integration

`engine/pointer_sequence.py` is opt-in and changes no existing login decisions. After the frozen run, `scripts/pointer_export_encoder.py` creates the portable TorchScript encoder and architecture/version/hash manifest. Optional torch/scikit-learn dependencies load only when their functionality is invoked.

Encode each recording separately. Enrollment uses the real account holder's own recording embeddings, with separate templates for each device class. Supply the chosen scoring method, threshold, and aggregation count explicitly; all sequence scores are **higher-is-better**, unlike the existing anomaly distance. A single block needs129 recorded coordinates (128 displacements); five contiguous blocks need641 coordinates. Insufficient data returns no verdict. Never aggregate probe blocks across recordings. Dataset-derived FAR is research evidence, not calibration of a live person's template or a single-click login guarantee.

Cosine and robust distance over the learned128-dimensional representation were predeclared as additional candidates while the 100epoch network was still training. `scripts/pointer_sapimouse_optimize.py` chooses among these and original OCSVM candidates using only twelve training-calibration users. It preserves the author-default comparator and matches aggregation length across reported methods.

## Frozen results and limitations

The training-only selector chose five-block cosine scoring over FCN embeddings. On24 dev identities, pooled EER fell from26.17% for the handcrafted baseline to13.87%; achieved FAR=1.66%, FRR=43.42% at the threshold calibrated for1% training FAR. The author-default FCN/OCSVM reference reached21.28% pooled EER. These protocols differ from published results and do not establish a SOTA improvement.

There are only 76 genuine aggregated dev decisions. Removing one participant at a time (as both claimant and probe source) gives selected EER 9.37–15.81%, FAR 1.14–1.94%, FRR 38.57–45.83%; those ranges measure cohort sensitivity, not confidence intervals. Five blocks require641 coordinates and took a training-calibration median16.89 seconds. Device/task confounds remain unresolved.

Both Balabit and SapiMouse have actual model-inference API replay artifacts, including exact score/decision agreement and sealed-test/loopback guards. See `pointer_balabit_api_replay.json`, `pointer_sapimouse_api_replay.json`, and `pointer_sapimouse_optimized/participant_deletion_sensitivity.json`. The child router is mounted in the isolated research API; no listener was launched during validation.

Pooled-versus-macro qualification: selected cosine's macro per-user EER is8.27%, versus7.73%for the author-default OCSVM and22.91%for the handcrafted baseline. Cosine improves the pooled calibration/operating point; it does not dominate the author reference on every metric.
