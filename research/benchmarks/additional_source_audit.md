---
title: Additional Motor and Paired Dataset Search
tags: [bioprint, datasets, provenance]
updated: 2026-09-20
---

## Touch motor evidence

The [official Google TSI release](https://github.com/google-research-datasets/tap-typing-with-touch-sensing-images)
provides target-key geometry and first-frame touch locations under CC BY 4.0.
This supports a target-normalized motor comparison. It lacks release events and
randomized keypad stimulus onset, so it cannot directly measure dwell or visual
reaction time. Four task blocks allow a preregistered within-session partition;
this does not establish cross-day robustness. A separate acquisition/benchmark
is being implemented; no result is assumed from dataset availability.

## Other candidates and access limits

[BehavePassDB's official repository](https://github.com/BiDAlab/MobileB2C_BehavePassDB)
requires a signed license emailed to the researchers before download credentials
are issued. No email was sent and no access agreement was signed. It is not an
immediately downloadable source for this run.

[KMT / CyberSignature](https://data.mendeley.com/datasets/fnf8b85kr6/1)
contains paired typing and pointer recordings from a payment-form task, a closer
task match than gameplay. The public metadata describes 88 card-owner files with
both genuine and other-person entries. This grouping needs careful actual-actor
and recording-duplication checks before an identity-disjoint protocol can be
claimed. Public landing-page and API requests from the execution environment
returned HTTP403; no recordings were acquired or parsed. The official demo
[repository](https://github.com/CyberSignature-EHU/CyberSignature) exposes one
negative-data file, not the full benchmark. That file was not downloaded as a
substitute for a representative dataset.

These candidates do not establish availability of an all-signal, jointly labeled
typing/pointer/device/keypad/bot corpus. Missing modalities must remain missing;
cross-dataset artificial identity pairing is prohibited.
