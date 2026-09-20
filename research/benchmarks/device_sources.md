---
title: Device and Multimodal Benchmark Sources
tags: [bioprint, datasets, validation, device, fusion]
updated: 2026-09-20
---

## Summary

Public longitudinal browser data **does exist**: the FPStalker authors release a 15,000-record subset. This updates the older availability note. Browser-instance linkage is not person verification or proof of a common physical computer. A genuine user can change browser/device; an impostor can use the same browser. Keep this axis advisory and separate.

## Source and access evidence

| Source | Evidence and access | Local use and limitations |
|---|---|---|
| [FPStalker authors' repository](https://github.com/Spirals-Team/FPStalker) | Two public archives; 143,126,719 bytes compressed; 257,121,411 bytes uncompressed SQL. Repository AGPL-3.0 license retained locally; no separate data license is stated in README. | `datasets/fpstalker`; stream without SQL execution/extraction. Public subset sampled from first half of original collection, not full paper dataset. Legacy browser/Flash era. |
| [How Unique Is Whose Web Browser? (2025) dataset README](https://github.com/aberke/fingerprinting-study/tree/master/data) | 8,400 consented participants; browser file at Harvard Dataverse, terms prohibit reidentification and redistribution. One row per participant. | Not downloaded: useful contemporary population distributions, cannot measure longitudinal genuine-user FRR by itself. |
| [BEACON (2026) official release](https://huggingface.co/datasets/beacon-gui/BEACON-Dataset) | CC-BY-NC-4.0; 28 participants, 79 sessions; separate keyboard, mouse, hardware files. Full release hundreds of GB exceeds budget. | Bounded paired prefix subset only. `datasets/beacon/split_manifest.json` pins revision and split; no video, packet captures or configurations fetched. Gaming interaction is a substantial domain shift from login. |
| [Buffalo CUBS](https://www.buffalo.edu/cubs/research/datasets.html) | Combined keystroke/mouse data requested by email. | Not downloaded; no unsolicited messages sent. |
| [ISOT combined dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/behavioral-biometric-datasets/) | Paired keyboard/mouse/site actions, 24 users; supervisor-signed agreement required. | Not immediately accessible; no contact initiated. |
| [HuMIdb](https://github.com/BiDAlab/HuMIdb) | Signed agreement and issued credentials; repository says over 5GB. | Access- and size-constrained; not downloaded. |
| [BeCAPTCHA-Mouse](https://github.com/BiDAlab/BeCAPTCHA-Mouse) | Real/synthetic trajectory benchmark; signed agreement and credentials required. | Relevant bot baseline but not available for immediate authorized direct download. |
| [DELBOT-Mouse](https://github.com/chrisgdt/DELBOT-Mouse) | Direct4.62MB archive in MIT-licensed repository; 3,551 human/bot traces. No separate data license identified. | Source/family-disjoint train/dev/test; GAN, phone and fast-human groups sealed. Device folders are not known person identities. |
| [Hedge2018 raw cognitive data](https://osf.io/cwzds/) | Authors publish repeated-task raw files; accompanyingREADME explicitly invites reanalysis. OSF node has no explicit machine-readable license. | Study 1 Stroop/Flanker, 47 repeat participants, two sessions. 2.04 MB training/dev subset; 10 test people not downloaded. No redistribution claim. |
| [BrainRun](https://zenodo.org/records/2598135) | CC0; 265 MB gestures/users/devices/games archive and separate 3.2 GB sensors archive. | Not downloaded. Published schema has game-level duration/accuracy/stage, not trial stimulus-onset RT; less direct than Hedge for cognitiveRT verification. |

## Methods and claims

[FPStalker](https://github.com/Spirals-Team/FPStalker/blob/master/algo.py) supplies a relevant learned linkage baseline: pairwise attribute similarities and a random forest. Our independent `fpstalker_inspired_random_forest` uses equality plus four string similarities, restricted depth/leaf size, and a different verification protocol. It is **not a reproduction of paper tracking-duration results**.

[ThresholdFP (IEEE Access, 2025)](https://research.sabanciuniv.edu/52205/1/ThresholdFP.pdf) is a newer relevant comparison. Its Algorithm 2 assigns larger mismatch weights to attributes that change less often within browsers. We implement those weights using training identities only, adapted to pair verification. Full accumulated lineage scores, candidate search and remediation are not implemented; do not label this a full ThresholdFP replication or a demonstrated SOTA result.

Histogram gradient boosting is a small contemporary tabular-model comparator, not a claimed published browser-biometric SOTA. Equal-weight attribute agreement and exact matching provide transparent controls. BioPrint's existing device score is evaluated on overlapping observed attributes only; absent fields stay absent. These inputs are narrower than the learned model's full dataset attributes, so comparisons to BioPrint combine algorithm and feature-availability effects.

The existing score's summed historical entropies are heuristic mismatch weights, **not independent bits of identifying evidence**. The [WWW 2024 study on fingerprinting risk](https://research.google/pubs/assessing-web-fingerprinting-risk/) specifically addresses correlations that invalidate naive entropy summation.

## Partition and evaluation protocol

FPStalker: deterministic SHA256 browser-ID split, 60% training / 20% development / 20% reserved test. The first metadata-only pass freezes IDs and counts before any feature tokenization. Test row bytes remain opaque in source archives; preparation explicitly skips decoding their feature fields. Evaluation supports only train/dev. Training identities split again into fitting and threshold-calibration groups. Each eligible browser has at least three observations; earliest is enrollment, at most 20 later observations are probes. Each probe has a genuine comparison and 20 deterministic sampled impostor enrollments. Dev identities were never used to fit global parameters. Dev enrollment is reference construction for unseen accounts.

Thresholds target training-calibration FAR 1% and 0.1%, with tied scores handled conservatively; report actual dev FAR/FRR and descriptive dev EER. Do not claim a fixed dev FAR was achieved merely because calibration targeted it. Trials share browsers and samples, so counts are not independent Bernoulli observations; uncertainty should be clustered by browser. No test-set results are produced.

BEACON: metadata-only selection of users with two paired sessions; sorted eligible IDs give seven training users, three dev users and two reserved test users. The first two sessions are enrollment/probe; single-session users are excluded. Initial 2MiB mouse prefixes were inadequate on **training** P002/S001 (typing started after the mouse prefix ended). Before dev inspection, revised uniformly to 16MiB and preserved v1 manifest. Align both modalities by observed timestamps and discard incomplete trailing CSV lines. Published size/hash metadata is stale for some corrected release files: receipts retain actual SHA256 and explicit mismatch status. This is a data-quality limitation, not permission to invent missing observations.

## Keypad, bot and joint-evaluation gaps

No verified open longitudinal dataset for this exact randomized-keypad task was found. Mouse data can validate motor features, but cannot validate visual-search timing or arithmetic difficulty slopes. Keypad run-level holdout is preferable to tap-level random splits because taps share a challenge/layout.

Existing bot checks are heuristics; synthetic replay/randomized attacks test implementation and known failure modes, not real-world bot FAR/FRR. Browser `isTrusted`, timing floors, and pixel-centre rules are not proofs of humanity or automation. Keep attack-family holdouts and genuine-data false-positive checks separate.

`delbot_benchmark.py` evaluates normalized trajectory geometry using source-heldout RF/logistic models and BioPrint's compatible pointer-rule subgroup. It does not manufacture browser probes, password events or trust flags. Its low-FAR failure on a new bot family remains in the report. The small dev set has 55 bot trials, so low-FAR precision is limited.

`cognitive_benchmark.py` compares condition-index RT slopes with condition means and richer RT/accuracy profiles across actual repeat sessions. Stroop/Flanker conditions are congruent, neutral and incongruent; labeling them 0/1/2 does not make them equal difficulty steps. Each session contains hundreds of trials; these results cannot validate a three-question login challenge.

The first cognitive experiment used 19 fitting participants, 9 training-calibration participants and 9 unseen development accounts. Session 1 is explicitly labeled **training enrollment support**; only session 2 is a validation probe. Descriptive dev EER was 48.61% for slope-only, 31.94% for condition means, 27.78% for the full RT/accuracy profile, and 30.56% for its learned random forest. At the threshold targeting 1% training-calibration FAR, the full profile produced 2/72 dev impostor acceptances and 7/9 genuine rejections. This is evidence against using this small cognitive battery as a standalone authenticator, with large uncertainty from only nine dev accounts. It does not establish the incremental value of cognition alongside typing.

Never fuse CMU user 1 with unrelated pointer/fingerprint user 1 and call it multimodal identity validation. BEACON permits genuine paired timing/motion fusion within sessions; hardware/context must remain separately reported. It still lacks browser fingerprints and this keypad task. An all-feature API exercise can establish integration correctness, but cannot establish all-feature biometric accuracy without appropriately paired ground truth.

## Reproduction

```bash
PY=/home/ananthakrishnan/miniconda3/envs/bigidea/bin/python
$PY scripts/device_download.py
$PY scripts/device_benchmark.py prepare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY scripts/device_benchmark.py evaluate
$PY scripts/beacon_download.py --download
$PY scripts/delbot_benchmark.py prepare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY scripts/delbot_benchmark.py evaluate
$PY scripts/cognitive_benchmark.py download
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY scripts/cognitive_benchmark.py evaluate
```

The benchmark imports repository-local `.research-deps`. Outputs: `device_results.json`, `device_dev_scores.npz`, dataset split manifests and download receipts. Raw live BioPrint recordings were not opened or modified.

Frozen learned models, input schemas and permitted dev feature matrices are in `research/benchmarks/{device,delbot,cognitive}/`. `test_saved_research_models.py` replays these artifacts against their reported rates. `device_uncertainty.py` provides 2,000-replicate probe-browser cluster intervals conditional on fitted models and enrollment references; it does not remove all dependence from shared impostor references.

## Related notes

- [[big_idea/05 Datasets and Data Availability]]
- [[big_idea/14 BioPrint Hackathon]]
- [[big_idea/15 Scrambled Keypad]]
