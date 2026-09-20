---
title: KMT Payment Dataset Access and Split Feasibility
tags: [bioprint, paired, blocked-access, sources]
updated: 2026-09-20
---

## Bounded discovery result

The [KMT author paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9679689/) describes paired behavior during payment entry, making it a closer task analogue than gameplay. The dataset is [DOI 10.17632/fnf8b85kr6.1](https://doi.org/10.17632/fnf8b85kr6.1). DataCite's public metadata confirms CC-BY-4.0 and 88 card-owner files containing 1,760 genuine/impostor instances. It does not report archive sizes or file-download identifiers. Metadata is saved in `datasets/kmt_metadata/datacite.response`.

Five-minute metadata-only discovery stopped with these concrete outcomes:

| Official endpoint/source | Outcome |
|---|---|
| `data.mendeley.com/datasets/fnf8b85kr6/1` | HTTP 403 |
| `api.data.mendeley.com/public-api/datasets/fnf8b85kr6/versions/1` | HTTP 403 |
| `api.data.mendeley.com/datasets/fnf8b85kr6/versions/1` | HTTP 403 |
| `data.mendeley.com/api/docs/` | HTTP 403 through direct HTTP client; web tool returns a loading shell |
| `api.datacite.org/dois/10.17632/fnf8b85kr6.1` | HTTP 200, 12,386 metadata bytes; sizes/formats empty |

The [older official Mendeley API guide](https://dev.mendeley.com/code/datasets_quick_start_guides.html) requires OAuth even for its public-dataset requests. No credentials were requested or bypassed. The [paper-linked author repository](https://github.com/CyberSignature-EHU/CyberSignature) has one `project/false_data/User0001.json` demonstration file (1,021,228 bytes in GitHub metadata), not the full 88-file release. The author organization exposes no other repositories. Neither this file nor any KMT measurements were downloaded. Metadata trees and access receipts are saved locally.

The paper's **231 MB describes the GUI application**, not the dataset. Therefore the dataset's <500 MB feasibility remains unverified. Do not substitute an unattributed third-party copy or claim that KMT has been evaluated.

## Required protocol before measurement access

A later check used the public `data.mendeley.com` host rather than the earlier
`api.data.mendeley.com` host. The landing page, public version endpoint, and
public file-list endpoint all returned HTTP 403 through the execution client.
The browser tool can display the landing description but supplies no file
inventory. `scripts/kmt_metadata.py` captures bounded metadata requests only;
`datasets/kmt_metadata/publichost1/` preserves exact URLs, timestamps, errors,
and executed source. No records were downloaded. These results establish an
environment-specific retrieval failure, not that the dataset is private or
universally unavailable. Do not repeat these endpoints without new evidence.

Each owner file combines its owner's genuine recordings with recordings by other people using that owner's assigned card. Thus splitting filenames alone does not establish performer-disjoint training/dev/test cohorts. It is not sufficient to relabel every negative example as its target owner's identity.

If public file metadata becomes available, first verify total bytes, license and publisher hashes. Inspect schema documentation only for actual performer IDs, recording/session IDs, collection order and duplication across raw/feature formats. Freeze an actor-to-record manifest without decoding timing or motion measurements. Assign actual people to train/dev/test using a deterministic hash; retain a record only when its performer and target owner belong to the same cohort. Otherwise an actor used as a training impostor may also appear as a dev/test genuine subject. This filtering may leave too few genuine or negative records; declare that limitation before model fitting.

Within permitted training actors, reserve distinct recording/session groups for fitting, model selection and threshold calibration. Assign chronological genuine recordings to enrollment support and later distinct recordings to probes. Use publisher recording IDs and opaque hashes to keep duplicate recordings and their derived features in the same role, rather than treating JSON and spreadsheet versions as independent samples. Keep keyboard/mouse streams paired by source recording and timestamps. Exclude entered card text from identity features.

If actual performer linkage is absent, a rigorous unseen-person split cannot be certified. The alternatives are a clearly labeled known-account evaluation with unresolved actor overlap, or deferring acquisition/evaluation. Neither is a substitute for the requested leakage-controlled benchmark. No split, training or test evaluation has been claimed in this access note.
