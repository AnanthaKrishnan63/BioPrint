---
title: Production Feature Coverage and Remaining Evidence Gaps
tags: [bioprint, benchmarks, limitations]
updated: 2026-09-20
---

## Scope and authority

This audit reads current implementation and saved dev reports, not live participant databases or reserved test measurements. Research inference replay establishes agreement with frozen offline models; it does not automatically replace or validate every production login decision. Source authority: `code/bioprint/engine/{device,bot,keypad,pointer,decide}.py`, research scripts and `ROOT 36 Problem Statements.pdf`.

## Coverage by signal

Current Type2Branch follow-up: the original fixed 1,000-event DEV protocol
remains infeasible and has no recognition metrics. A separate, disclosed
one-capture protocol is frozen in `type2branch_capture_protocol_v1`: all 79
active identities remain in coverage, eligible captures use exact 25/50/75/100
event prefixes, and shorter captures request additional verification. Both
equal-budget TRAIN-only arms completed; the frozen rule selected mixed
epoch4/500updates. Calibration and DEV evaluation are complete: coverage79/79,
FAR9.13664%, FRR30.37975%, pooled discrete EER13.93216%. The actual isolated
API worker reproduced all6241scores and every metric exactly. The1% calibration
FAR target failed cross-session transfer. Prior DEV exposure and documented
preprocessing adaptations remain; this is not a pristine final test or exact
published-protocol reproduction. These additions do not resolve the all-signal
paired-data gap below.

The later paired BEACON Type2Branch transfer preserves140windows (81below25
events) and all three DEV people. Its hybrid FAR24.69% improves on the matched
old hybrid's45.68%, but FRR worsens from30.86% to67.90% and EER from38.27% to40.74%.
Eight comparators use the same243claims and disjoint TRAIN fitting, selection,
and calibration identities. The result does not establish an overall improvement
or resolve gameplay-to-login transfer. Actual API replay reproduced 1,944 scores
exactly, with zero decision changes at all three thresholds and identical
pooled/per-person metrics. API scoring uses offline encoder features; prior DEV
exposure and the missing all-signal corpus remain material limits. The latest
consolidated evidence is `summary_v11.json` (72 DEV rows, 25 source reports and three
separate infeasible experiments).

| Production signal | Real-data evidence obtained | Missing evidence / interpretation |
|---|---|---|
| Keystroke timing | CMU, KeyRecs and learned sequence benchmarks; see keystroke reports | Fixed-password and free-text protocols differ; performance does not establish cross-password, mobile keyboard or accessibility robustness. |
| Pointer behavior | Balabit/SapiMouse trajectories; BEACON paired mouse/keyboard | Generic trajectory windows do not validate all 14 submit-target features, especially key-to-movement latency, target-normalized landing, hover, settling and Fitts normalization. Source tasks/devices can correlate with identity. |
| Device fingerprint | FPStalker longitudinal browser-instance verification, 154 dev browser IDs | Not a person or physical-computer identity label. No shared-laptop impostor ground truth or controlled same-machine browser-switch experiment. |
| Keypad cognitive | Repeated Stroop/Flanker RT and accuracy from nine unseen people | Transfer experiment only: no scrambled layout, target geometry or digit entry. Neither arithmetic difficulty slope nor exact keypad features validated. |
| Keypad motor | General pointer evidence and Google TSI target-relative touch landing/timing | TSI validates a keyboard-task motor proxy, with poor dev results; no dataset couples randomized digits, tap/release, cell rectangles and repeated mouse/touch sessions. Dwell and cross-input-class transfer remain unvalidated. |
| Bot detection | DELBOT human/bot trajectories, held-out acquisition source and bot family | Evaluates geometry classifier and only production pointer-rule subgroup. Does not validate browser trust, automation globals, key-count, timing, keypad or replay rules. |
| Joint verification | BEACON simultaneous keyboard/mouse plus coarse hardware; three dev people | Actual paired evaluation, but gameplay rather than login. First fusion model did not improve dev EER; do not combine unrelated datasets and call that multimodal validation. No all-signal dataset including cognitive and bot attacks. |

## Device and context coverage details

The production scorer compares 22 attributes with hardware/browser/network groups. Its FPStalker overlap adapter supplies UA (family/version), platform, languages, canvas hash, GPU renderer/vendor (including derived family), cookies, DNT and screen dimensions/color depth where present. HTTP language versus browser language and source formatting remain collection differences.

It does **not** supply installed fonts, timezone-country, CPU cores, device memory, touch count, usable screen, UA brands, client-hints platform, plugin count, audio fingerprint or IP/network. Source timezone offsets cannot be honestly mapped to IANA timezone-country; source plugin-list hashes are not plugin counts. Source Flash fonts are not the live font probe. Missing attributes are skipped, never filled with made-up equality. The learned fingerprint models additionally use source storage flags and browser/OS fields: comparison with the overlap heuristic changes both representation and method.

IP is deliberately excluded from this public-data experiment. Shared campus NAT, moving networks, VPNs and multiple browsers require separate contextual validation. Published per-attribute entropy weights in the current engine are heuristics; correlated attributes do not establish additive independent identity information or calibrated probabilities.

## Cognitive, bot and replay limits

Production cognitive vectors contain interval, interval-per-Fitts-bit, hold and hold fraction. These mix visual search, reaching, pressing, input latency and distraction; they are not isolated measurements of cognitive ability. Production motor vectors contain 11 mouse or three touch dimensions. No acquired source measures all of these in the intended keypad flow.

The cognitive condition-index slope is an arbitrary contrast across congruent/neutral/incongruent tasks, not three equally spaced arithmetic difficulty levels. Hundreds of trials per session differ materially from a short login. Dev EER was 48.61% for slopes versus 27.78% for full RT/accuracy profiles, with poor low-FAR rejection rates. This does not prove that full cognitive profiles add value to typing or that one login challenge outperforms another. See `cognitive_uncertainty.json` for participant deletion sensitivity, explicitly not confidence intervals.

DELBOT supplies no authentic browser `isTrusted`, webdriver flags, password entry counts or prior-keystroke replay labels. Those fields were not fabricated for classifier input. The engine's 4 ms replay threshold has an existing CMU rationale, but this work has not measured a held-out labeled attack corpus of browser replay, jittered replay, OS injection and adaptive bots. Offline API replay means reproducing model predictions; it is **not** an adversarial replay-defense evaluation. Rules such as a sub-120 ms keypad search or pixel-perfect landing need observed genuine distributions, including assistive input, before universal-human claims.

## Judging brief and remaining demonstration work

Reliability carries 25 points: report repeated genuine/impostor FAR/FRR with frozen training thresholds and uncertainty. Novelty carries 15 points: name reproduced architectures and adaptations honestly; do not label every classifier SOTA. Latency carries 10 points: API numerical parity does not establish end-to-end collection/login latency. Ease of use and customer satisfaction each carry 10 points: long-task research data does not establish a comfortable enrollment/login experience.

The brief requires a genuine login, same-password impostor or bot block, distinct automation signal, and no OTP or secondary-verification fallback. Current decision code can request extra typings or keypad checks. Those are additional behavioral collection, but compliance with the brief's broad wording remains a documentation/demo question, not a claim this benchmark resolves. Repository instructions favor additional verification; do not silently change that policy while auditing. The report, runnable app, setup instructions and live demonstration remain separate deliverables from statistical benchmarks.

## Dataset budget audit

The current metadata-only snapshot, `dataset_budget_capture_api_complete.json`,
counts **1,210,656,046 bytes total** and a conservative largest-source bound of
**468,601,109 bytes**, including temporary fixtures and other derived arrays.
Both requested limits pass. The tables and older snapshots below are historical;
the script's documented exclusions include model weights and dependencies.

Filesystem sizes below were measured with directory metadata (`du -sb`), without opening dataset records. Snapshot totals include archives, metadata and derived caches in each listed data directory; model artifacts and Python package caches are separate.

| Data directory | Bytes |
|---|---:|
| `datasets/beacon` | 326,858,593 |
| `datasets/fpstalker` | 153,601,507 |
| `datasets/delbot` | 5,109,068 |
| `datasets/cognitive` | 2,539,511 |
| `datasets/brainrun` (metadata only) | 5,219 |
| `data/benchmarks/balabit` | 84,709,166 |
| `data/benchmarks/sapimouse` | 35,157,313 |
| `research/benchmarks/data/keyrecs` | 31,088,187 |
| `code/bioprint/eval/data` (CMU) | 4,670,011 |

The listed total is **643,738,575 bytes (0.644 GB decimal)**, below 5 GB, and each directory is below 500 MB. FPStalker's separately measured logical SQL payload is 257,121,411 bytes; DELBOT's logical archive payload is 17,498,093 bytes. Reserved records may exist inside downloaded public archives, but split-aware loaders do not decode them. BEACON and cognitive reserved files were never fetched. These statements distinguish downloaded archive presence from feature inspection and test evaluation.

Related notes: [[device_sources]], [[device_worklog]], [[keystroke_sources]], [[pointer_sources]].

Subsequent acquisition: Google TSI adds about 28.5 MB and target-relative touch
motor evidence. The reproducible metadata-only audit in
`dataset_budget_20260920.json` totals 672,818,713 bytes across ten dataset
directories, still within both limits. Run `scripts/dataset_budget.py` for the
current snapshot. The earlier table is a historical acquisition snapshot.

Latest expanded accounting includes derived feature/score arrays stored outside
raw data directories. `dataset_budget_final_integration.json` records
689,405,060 bytes (689 MB), with every dataset below 500 MB and the total below
5 GB. This metadata-only audit conservatively counts profile NPZ files too.

The later per-account CMU experiment adds stored dev score matrices. Updated
`dataset_budget_account_selection.json` totals 695,357,358 bytes (695 MB),
still within both dataset limits. Earlier snapshots remain preserved.

The FRR-directed CMU follow-up and source-discovery metadata bring the recorded
total to 703,308,097 bytes in `dataset_budget_far_selection.json`. Derived score
matrices are included; both storage limits still pass.

HMOG selective acquisition is now complete for 12 preregistered identities,
with only four fit identities' TRAIN sessions subsequently audited.
`dataset_budget_hmog_complete.json` records the acquisition-time snapshot of
1,149,223,443 bytes overall (1.15 GB), including temporary test/replay fixtures.
HMOG is charged all shared discovery metadata; even assigning every temporary
fixture and unattributed report to the largest dataset leaves it below 500 MB.
Session roles are frozen and each TRAIN/DEV stage has now run under its guard;
test observations remain sealed. Strict extraction v2 found zero usable windows.
Separately frozen retained-pair v3 supplied207fit-cohort windows and an800-update
encoder experiment. Original calibration lacked one account and remained failed;
the explicit fallback uses encoder-seen identities' disjoint enrollment sessions.
DEV has galleries for four accounts but probes for only two:31/154 inspected
candidate windows,20.13% coverage. Full-cohort status remains infeasible. Conditional
joint EER45.16%,FAR0%/FRR100% at the1% TRAIN target and372-score API parity are
negative observed-window evidence. No all-feature or complete-cohort claim follows.
