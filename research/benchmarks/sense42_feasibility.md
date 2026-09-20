---
title: SENSE-42 Paired Behavioral Data Feasibility
tags: [bioprint, datasets, provenance, fusion]
updated: 2026-09-20
---

## Finding

SENSE-42 is a newly audited candidate, not an acquired benchmark. The
[authors' repository](https://github.com/Catherine9811/HCI-SENSE-42) describes
42 people completing two-hour desktop sessions with synchronized keyboard and
mouse data. This may support paired behavioral analysis, but one continuous
session does not establish cross-day verification. Exact BioPrint keypad,
browser fingerprint, and bot-attack coverage remains unverified.

## Access and size evidence

The [official Zenodo record](https://zenodo.org/records/20328099) marks files
restricted and requires an approved research-access request. Its public API
returned HTTP 200, `access_right=restricted`, and an empty file list. A public
README's description of availability must not be treated as download permission.
No access request, agreement submission, or participant-file retrieval occurred.

The record links [Figshare metadata](https://api.figshare.com/v2/articles/29971873).
This API returned 42 files totaling **56,176,822,056 bytes**, with the smallest
**1,259,822,664 bytes**. Its title and description identify cleaned, downsampled
EEG, not the desired keyboard/mouse recordings. Even one complete file exceeds
the 500 MB source limit. No measurement download links were followed. Figshare's
CC0 designation concerns that EEG release; it does not override Zenodo access.

Figshare additionally links Synapse `syn68713182/wiki/633562`. Although the
browser page failed, a subsequent anonymous official REST metadata audit
succeeded. The project contains `Behavioural` (`syn68714674`), with `CSV`
(`syn68714788`) and `PsyDAT` (`syn68714789`) folders. Each listing returns all
42 file entities with no next-page token. This verifies the inventory, not
record completeness or permission to download.

The first listed CSV, `syn68723176`, returned HTTP 403 for `/filehandles`.
Its `/permissions` response explicitly reports `canView=true`,
`canDownload=false`, `isCertificationRequired=true`, and `isEntityOpenData=false`.
The [official permissions documentation](https://rest-docs.synapse.org/rest/GET/entity/id/permissions.html)
identifies this calculated permission endpoint as authoritative over an ACL-only
interpretation. An empty `/accessRequirement` list therefore does **not** mean
anonymous download is authorized. No file download was attempted. The checked
file is not an anonymous acquisition route; permissions on every other file
were not individually tested. File sizes remain unverified. Authenticated access
would need to be established before further acquisition work on this route.

## Reproduction and next gate

`bash scripts/research.sh scripts/sense42_metadata.py <uniqueAlphanumericLabel>`
fetches fixed metadata/documentation URLs, bounded at 2 MB each,
with 20-second timeouts and exclusive output directories. Inspect receipts:
HTTP errors are recorded, not interpreted as successful retrieval. The raw
README URL using `main` returned 404. The follow-up correctly used `master`
and saved its successful HTTP 200 response.

`datasets/joint_feasibility_metadata/sense42_v1/receipt.json` preserves sandbox
DNS failures. `sense42_v2/` preserves successful official metadata responses,
SHA256 receipts, and the original README error. Additional exclusive directories
`sense42_{synapse1,synapse2,behavioral1,formats1,handle1,permissions1}` preserve
the successive metadata checks. Reproduce these stages with the corresponding
`--synapse`, `--behavioral`, `--formats`, `--handle`, or `--permissions` flag and
a fresh label; POST child-list requests are read-only metadata operations.
No observations were parsed, so no new FAR/FRR/EER or split claim follows.
Verified download authorization, bounded sizes, an acquisition plan, and frozen
identity/session partitions are required before
any measurements may be read. Related notes: [[additional_source_audit]],
[[coverage_gaps]].
