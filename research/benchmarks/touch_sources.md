---
title: Touch Landing and Timing Transfer Benchmark
tags: [bioprint, touch, sources, limitations]
updated: 2026-09-20
---

## Source and relevance

[Google's Tap Typing with Touch Sensing Images dataset](https://github.com/google-research-datasets/tap-typing-with-touch-sensing-images) accompanies Lertvittayakumjorn, Cai, Dou, Ho and Zhai, [Can Capacitive Touch Images Enhance Mobile Keyboard Decoding?](https://doi.org/10.1145/3654777.3676420), UIST 2024. The publisher licenses the data CC-BY-4.0. The original task predicts intended keys; it is not a biometric identity benchmark, so this work does not reproduce biometric SOTA.

The release has 16 people, four typing blocks per person, and 43,735 aligned taps. Collection used two Pixel 6 Pro phones. Blocks belong to one visit with short breaks, not separate days. The sample excludes people with relevant vision/motor impairment, limiting accessibility generalization. Only first-frame touch geometry is recorded; release time and stimulus onset are absent.

The 28,332,692-byte touch CSV and 5,409-byte keyboard layout were downloaded. No prompts or images were needed. Publisher Git blob SHA1 and local SHA256 receipts live in `datasets/touch_tsi/split_manifest.json`. Raw CSV SHA256: `f6aaa9c79596a5d61c5d2429d722db30013fd92d333d28863adc77e54ef4d8a3`. Directory storage including metadata is 28,382,910 bytes at this snapshot, before the small role manifest.

## Frozen protocol and features

`touch_tsi/preregistered.json` was saved before measurement download. Accounts user01–12 use task1 trials 0–14 for enrollment and 15–29 for fitting probes. Task2 trials 0–14 select the global model; trials 15–29 calibrate thresholds. Task3 supplies dev probes. All task4 rows and all user13–16 rows are sealed. These are returning enrolled accounts, not unseen identities. Metadata prefix checks precede CSV measurement decoding; 19,140 reserved rows remain opaque. Hashing downloaded bytes is integrity verification, not feature inspection.

Each eligible trial becomes six target-normalized landing summaries (mean, standard deviation and median for horizontal/vertical offsets), plus four intertap summaries. Trials need five aligned taps and three positive gaps no longer than five seconds. Key labels only retrieve target geometry; neither prompt content nor label frequencies enter identity features. Gaps stay within trials. Timing measures intertap cadence, not reaction time or hold. No capacitive heatmaps or ellipse measurements are used.

The landing baseline uses six dimensions and a training-scaled absolute distance. The selected model uses ten dimensions including timing. Consequently, its comparison combines added features and classifier changes; it does **not** isolate an algorithm improvement. A balanced logistic model and depth-six random forest were compared using identical ten-dimensional inputs on training selection only. RF won (34.07% versus 35.34% selection EER), then remained frozen. No model was refitted on dev.

## Results and limits

There are 356 eligible dev trials from 12 accounts, producing 356 genuine and 3,916 impostor comparisons. Three observed dev trials failed preregistered eligibility; one expected trial has no source rows. The source also has two training eligibility exclusions. No exclusions were chosen from model performance.

| Method | Dev descriptive EER | Dev FAR at training 1% FAR threshold | Dev FRR |
|---|---:|---:|---:|
| Six-feature landing baseline | 36.75% | 0.996% (39/3916) | 95.51% (340/356) |
| Training-selected ten-feature RF | 34.45% | 0.996% (39/3916) | 91.85% (327/356) |

This does not establish a usable login verifier. Comparisons share participants, trials and reference templates; 3,916 is not an independent population sample count. Differences are descriptive, not evidence of statistically significant improvement. Timing, task and posture effects remain possible. Landing offsets offer real evidence relevant to two production touch-motor dimensions, but hold is missing and no randomized keypad task is present. Cognitive and joint-login claims remain unsupported.

`touch_tsi/api_replay.json` verifies 4,272 pair scores through the isolated ASGI API, with maximum offline discrepancy 3.33e-16 and identical per-row decisions and reported FAR/FRR. No listener was started and no production database was used. The research model is available for reproducible inference, not installed as a production authentication decision.

## Reproduction and worklog

Use the bigidea interpreter and two CPU threads:

```bash
python scripts/touch_benchmark.py download
python scripts/touch_benchmark.py prepare
python scripts/touch_benchmark.py train
python scripts/touch_benchmark.py evaluate
PYTHONPATH=.research-deps:code/bioprint python scripts/modalities_api_replay.py touch_tsi
python -m unittest discover -s scripts -p test_touch_benchmark.py
```

Training/dev commands refuse to overwrite frozen runs. Keep original artifacts when conducting new preregistered experiments. Acquisition first stalled, then succeeded through the same raw publisher URL with a download query; both source files verified against immutable Git object metadata. Three synthetic tests passed for reserved-row guards, target normalization and within-trial gap handling. Models, schema, enrollment templates, calibration matrices and dev matrices are saved under `research/benchmarks/touch_tsi/`.

Related notes: [[coverage_gaps]], [[device_sources]], [[device_worklog]].
