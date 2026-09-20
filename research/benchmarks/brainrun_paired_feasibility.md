# BrainRun paired extension feasibility

Audit date: 2026-09-20. Identifier audits and a bounded TRAIN-only schema
attempt have executed; no models fitted or DEV/test behavior decoded.

## Acquisition and identity status

The compressed archive is now acquired and verified against the published MD5
and the independently inspected ZIP directory. SHA256:
`070f43c7a068e96e484d303ad67e85d018859aebbf1c25d6e5b7fe563426bb6e`.
No full extraction was performed. The post-acquisition budget snapshot is
`dataset_budget_brainrun_acquired.json`: total1,477,025,851bytes and conservative
largest-source bound468,701,132bytes, both within limits.

Identifier-only inspection of devices found2,418records,2,404device IDs and
2,195user IDs. Two device IDs each belong to three users. The failed initial
one-owner assumption remains recorded in `identity_roles_v1`. Frozen v2 roles
exclude both shared devices and all six affected users before deterministic
active selection:120fit,40selection,40calibration and60DEV. Remaining users,
including every reserved test identity, remain sealed.

The first bounded eight-fit-user schema attempt stopped on a games record whose
device ID is absent from the device table. No owner was inferred. Its failed
plan is preserved in `brainrun_train_schema_v1`; the subsequent metadata-only cross-collection
identifier audit completed106,805games and3,110,101gestures. It found30game
records and2,326gesture records with unmapped devices, zero missing device IDs
and zero conflicts with known mappings. Frozen v3 adds the20observed orphan
device IDs to explicit quarantine without changing any identity roles. A
separate bounded eight-fit-user schema audit is running with this overlay. The count of authorized fitting documents
decoded before that failure was not persisted and must not be claimed as zero.
No selection/calibration/DEV/test behavioral documents passed the decoder gate.

The [author paper](https://doi.org/10.3390/DATA4020060), Tables 1, 5 and 6,
documents gesture device/session/time fields, device-to-user mapping, and
user/device/game-time fields with task, stage and correct/wrong totals. A
potential join is the same user/device and a uniquely containing game interval,
cross-checked against the gesture screen. This is an inferred join, not a
documented direct gesture-to-game foreign key. Ambiguous intervals cannot be
silently assigned. Application restart sessions do not establish separate days.

The [Zenodo release](https://zenodo.org/records/2598135) is CC0. Its non-sensor
archive is265,000,157bytes, published MD5a098d5a4028f38a97841c40470ca73e8.
The separate3.219GB sensor archive exceeds the source budget and is excluded.

## Executed metadata-only acquisition

`bash scripts/research.sh scripts/brainrun_archive_metadata.py` completed using
two exact HTTP206 ranges:22-byte ZIP footer and1,170-byte central directory.
No member payload was requested or decompressed. The immutable receipt is
`datasets/brainrun/archive_metadata_v1/report.json`; directory SHA256 is
`840df2527d6ed005b3e4176c0ce6107de188db320a4c35fc2da8606b9aef6d8a`.

| BSON member | Compressed bytes | Expanded bytes |
|---|---:|---:|
| devices |93,335|403,209|
| games |2,734,387|29,837,454|
| gestures |261,525,430|1,480,511,108|
| users |644,561|22,341,646|

Total expanded archive size is1,533,094,281bytes. Full extraction would violate
the500MB source limit. Keeping the compressed archive and streaming a bounded
derived subset may fit, but requires a separate acquisition/split protocol.
The directory does not partition individual participants into separate members.

## Required next gate

Freeze identity assignment from identifier metadata before decoding behavior.
Implement a bounded BSON scanner that can inspect permitted identity/join
metadata, skip sealed documents without decoding their behavioral values, and
avoid writing the full expanded stream. Verify those guarantees with generated
poisoned sealed records. Budget compressed data, selected derived records and
temporary files together. Only then audit TRAIN repeated sessions and unique
gesture/game joins. Do not choose participants using DEV behavior or claim
eligibility from the schema alone.

This candidate can add genuinely paired touch motor behavior, aggregate game
performance and coarse device context. It does not supply typing, browser
fingerprints, labeled bots, randomized-keypad geometry or stimulus-onset reaction
time. Device identifiers are join keys, never behavioral predictors. It cannot
by itself complete the all-feature objective or establish a cognitive SOTA
baseline. No new recognition metrics exist for this source.
