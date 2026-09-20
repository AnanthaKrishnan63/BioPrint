---
title: Three-Level Arithmetic Response-Time Source
tags: [bioprint, cognitive, datasets, provenance]
updated: 2026-09-20
---

## Source and task

The [author paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174949)
describes arithmetic correctness judgments at three difficulty levels: single-digit
addition, two-digit operations with carrying, and mixed operations. Participants
answer with left/right keys. Each level has five blocks of five experimental
trials, following practice; trials have a 2.5-second deadline. Levels occur in
fixed low-to-high order within one visit. Thus difficulty is confounded with
order, and there is no cross-day authentication evidence. Misses/censoring must
not be discarded silently. This is closer to the proposed arithmetic contrast
than Stroop/Flanker, but does not measure scrambled-keypad visual search.

The paper explicitly supplies data through the
[author repository](https://github.com/cityuCompuNeuroLab/MentalWorkload_cognitiveEEG).
Its README describes `imageT`, `keyT`, `CorrectAns`, difficulty labels `l/m/h`,
and rounds T1 (practice), T2–T6 (experimental). The paper describes 20 people;
the complete retrieved GitHub tree contains **19 subject ZIPs**. Do not invent
the missing twentieth participant. No separate repository data license was found
in the complete tree; the paper is CC BY and explicitly offers research access.
Do not infer an independent data redistribution license from that alone.

## Frozen acquisition and analysis gates

Tree `d0f32306aeb6b355c5e0d99997f9c990d60bdf82` lists 48,833,317 archive bytes.
Before any archive access, SHA256(`20260920:<archive name>`) ordering assigned
seven TRAIN-fit, four TRAIN-calibration, four DEV, and four sealed-test identities.
Only the 15 non-test ZIPs are authorized for acquisition: **38,515,916 bytes**.
Publisher Git blob hashes and exact lengths are verified; ZIP directories may
be inspected but measurement members remain sealed at this stage.

The acquisition plan assigns T2–T3 as enrollment support and T4–T6 as probes,
according to identity role; T1 is excluded. Only `Cal` arithmetic is proposed.
A separate TRAIN-only schema audit and frozen feature/model protocol are required
before decoding measurements. Ordinal slope is an explicit contrast, not a
claim that difficulty steps have equal physical magnitude. All error/timeout and
missing-level handling must be defined on TRAIN before DEV access. Few people
and probes will severely limit low-FAR precision. This source cannot supply
all-app-feature fusion or establish a SOTA identity-verification result.

Scripts: `arithmetic_source.py` captures metadata; `arithmetic_acquire.py prepare`
freezes roles and `acquire` fetches only non-test archives. Use
`bash scripts/research.sh`. Metadata and plans are in `datasets/arithmetic_metadata`
and `datasets/arithmetic`. Neither acquisition nor schema documentation is
recognition validation. Related: [[coverage_gaps]], [[device_sources]].

## Verified acquisition result

Acquisition completed with all 15 allowed archives, exactly 38,515,916 bytes.
Independent local SHA256 revalidation passed every file; all four test archives
remain absent. Directory metadata contains all 15 expected non-practice `Cal`
members per acquired participant. This does not prove valid measurements.
No ZIP measurement members have been decoded. `schema_plan_v1.json` freezes the
next audit to 105 members from the seven TRAIN-fit identities only.

Seven acquisition/budget tests passed. `dataset_budget_arithmetic_acquisition.json`
records 1,207,780,628 total bytes and a conservative largest-source bound of
465,746,016 bytes, both within the requested limits. Metadata is charged to the
same arithmetic source. Temporary-fixture retention differs from older snapshots.
