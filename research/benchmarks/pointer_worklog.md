# Pointer research worklog

## 2026-09-20 — Initial audit and protocol

- Inspected AGENTS.md, the production pointer extractor and scorer. Production pointer needs real target geometry; Balabit does not supply it. Established explicit domain/feature limitations instead of fabricating targets.
- Researched publisher datasets and primary method sources (see pointer_sources.md).
- Retrieved only GitHub tree metadata initially; fixed whole-session split before opening measurements. Metadata lists 65 genuine training sessions / ten users / 99,342,740 bytes.
- Created pointer_download.py: official test excluded; reserved test sessions not downloaded; source commit pinned; train/dev download concurrent with bounded workers.
- Created pointer_benchmark.py: 29 non-position trajectory descriptors, actual production scorer math baseline, RF/ExtraTrees supervised comparison. All tuning and thresholds use train-internal heldout sessions; dev load occurs only after frozen choice is written.
- Known limitations: random whole-session partition (timestamps do not establish longitudinal ordering); ten identities; correlated windows; no device labels; current deep SOTA reproduction and production API integration remain outstanding.

## 2026-09-20 — Verification before training

- Four synthetic-only checks pass: test-split access fails before opening paths; FAR threshold handles tied scores conservatively; feature vectors ignore coordinate translation; duplicate timestamps are rejected safely.
- Runtime dependencies are provided by the parent task in repository-local `.research-deps`; `bigidea` Python is used throughout. First import attempts exposed missing cloudpickle/narwhals dependencies, subsequently resolved by the parent; no measurements were used in those checks.
- Added exact publisher Git blob verification for every train/dev CSV read. Added bounded curl connection/transfer timeouts for future download runs.
- Export fitted selected per-user models, baseline centers/spreads, frozen calibration decisions, dev scores, and both pooled and per-user metrics for independent audit/replay.

## 2026-09-20 — Balabit experiment B1 (completed; unsatisfactory)

- Download completed: 84,142,665 bytes of permitted train/dev CSVs. Ten reserved test sessions and all official test files remain undownloaded.
- Frozen selection: ExtraTrees, leaf size 1, 160 trees per claimed user. At training-calibrated 1% FAR target, dev FAR=1.933%, FRR=92.446%, pooled descriptive EER=56.183%, AUC=.4711. Existing scaled-Manhattan math: FAR=1.888%, FRR=98.201%, EER=48.606%, AUC=.5526.
- Macro per-user EER differs from pooled EER: ensemble 30.094% versus baseline 38.296%. This difference exposes poor between-user score calibration. No general improvement claim is justified.
- Coverage is limited: 1,206 fitting, 212 calibration, 278 dev windows; only 26 fit / 7 calibration / 9 dev sessions yield usable continuous windows. Nine of ten identities evaluated. Event-window continuity rules are unsuitable for much of this collection; discarded sessions/identities must not be hidden.
- Saved results.json, frozen_training_selection.json, per-user fitted models and dev score archives. No optimization will use these dev outcomes. New SapiMouse experiment is independently grounded in published architecture and data protocol.

## 2026-09-20 — SapiMouse S1 (training started)

- SapiMouse archive is 8,056,327 bytes; all CSVs total 33,433,632 bytes. Downloaded, listed archive metadata, and froze identity split before reading measurements. Extracted only train/dev subjects; 24 test identities remain sealed in the ZIP.
- Retrieved exact author architecture, wrapper, normalization, OCSVM, settings and license sources; verified source Git blob hashes against publisher tree metadata. Found published probe-dependent score normalization and impostor aggregation across users, which our protocol forbids and documents as deviations.
- Implemented actual FCN architecture in PyTorch CPU, 128x2 inputs, author filter/kernel dimensions, BN/ReLU/GAP, default OCSVM, training-only parameter/threshold comparison. Three synthetic architecture/protocol tests pass.
- Live training handle 10199; log pointer_sapimouse_run.log. Training uses3,501 windows/60 identities; checkpoint selection uses1,119 second-session windows from those same training identities. Separate12 training identities calibrate verification. No dev measurement read before frozen selection; no test contents read at all.
- Archive metadata audit found245 CSV files: user35 and user62 have4 each, user85 has3. The others have2. Protocol is precisely suffix-based: all `_3min.csv` files form fitting/enrollment, all `_1min.csv` files form session-heldout probes. This is not a verified date ordering. No window or score aggregation crosses file boundaries.
- First3 FCN epochs completed in52 seconds. A full100epoch run is expected to take approximately29minutes on2CPUthreads; this is an estimate, not a completion claim.

## 2026-09-20 — Support/training role clarification before dev analysis

- Clarified the distinction between an unseen-to-encoder subject cohort and record-level split categories. For the 24 unseen dev-cohort subjects, `_3min.csv` files are explicitly **TRAINING support enrollment** (`role=dev_support_enrollment`, excluded from global representation fitting); `_1min.csv` files are **DEV probes** (`role=dev_probe`, never fitted).
- Canonical manifest changed before any dev-cohort measurement analysis, with no change to file membership, algorithm or thresholds. Previous cohort-only manifest and executing script are retained for audit. The live100epoch process already held the original cohort labels in memory; its fixed3min-enroll/1min-probe selectors access the identical canonically classified files. Reproducible scripts now reload the role manifest and assert support/probe split/suffix guards.
- New latent cosine and robust-scaled-distance variants were predeclared before dev analysis in optimization_plan.json. Their comparison script will choose only by training12-user calibration, including all original OCSVM candidates, then validate the frozen choice on dev probes. No dev-based score normalization is allowed.
- Optional engine/pointer_sequence.py now supports OCSVM, cosine and latent robust distance with lazy dependencies and explicit thresholds. Four generated-data runtime tests pass. No live login behavior or user database changed.
- Added train/serve preprocessing parity check using generated CSV coordinates and the application encoder transform; all 4 Sapi protocol/architecture checks pass. Runtime module has4 generated-data tests passing. Split audit verifies171 training files,24 dev probe files,50sealed Sapi files unextracted; Balabit ten sealed files undownloaded.
- Published source code is retained with its Apache-2.0 license. Dataset publisher pages provide public research downloads; the code license should not be assumed to grant separate dataset redistribution rights. No datasets are published or sent externally by this work.

## 2026-09-20 — Pointer API integration and Balabit replay

- Added research_pointer_api.py router and pointer_api_replay.py. The parent mounted it only in the isolated loopback research API. No network listener or production-database access was started.
- Balabit replay ingests278 actual dev feature vectors from the dataset endpoint and recomputes both models' scores for9claimed identities. All scores match saved validation artifacts exactly (maximum difference 0); threshold decisions, FAR and FRR also match.
- Verified sealed-test access 403, non-loopback access 403, cross-origin403, malformed arrays 422, and finite values exceeding float32 range422.
- Initial sync FastAPI routes hit a confirmed sandbox threadpool wait (PID 543159, epoll/futex); stopped only that owned replay process, converted pointer route functions to async, and reran successfully. Neural training handle 10199 was never interrupted. Replay success artifact: pointer_balabit_api_replay.json.

## 2026-09-20 — SapiMouse S1/S2 completed

- All 100 FCN epochs completed. Training-session holdout selected epoch 27; model has273,468 parameters. No dev/test measurement selected that checkpoint.
- Initial trained-default optimization selected enrollment-normalized OCSVM with five blocks. Dev pooled EER 18.364%, FAR 2.403%, FRR 61.842% at a training-calibrated1% FAR target. Author-default raw OCSVM pooled EER 21.281%, FAR 2.231%, FRR 80.263%.
- Predeclared latent alternatives selected **cosine@5**, solely by the twelve training-calibration users. Dev pooled EER 13.873%, AUC 0.9488, FAR 1.659%, FRR 43.421%. Same-observation handcrafted/production-scorer baseline: EER 26.173%, AUC 0.8238, FAR 8.295%, FRR 60.526%. This is development evidence, not a current-SOTA or production-security claim.
- Evaluation:24 unseen-encoder dev subjects;1,359 training-support enrollment blocks;427 dev probe blocks;76 genuine and1,748 impostor comparisons after five-block aggregation. Test 24 subjects remain sealed.
- Five blocks need640 displacements/641 coordinates. Training-calibration median observed duration16.89s, p10–p90=8.57–33.62s; calibration has only 35 genuine aggregated decisions. Not comparable to a single reach-and-click.
- Participant-deletion sensitivity removes each identity as both claimant and probe owner, holding thresholds fixed. Selected FAR range1.136–1.939%, FRR 38.571–45.833%, EER 9.371–15.809%. These are cohort-sensitivity ranges, not confidence intervals. Computation uses saved scores only, with no model tuning or new raw data reads.
- Exported encoder.torchscript.pt and version/architecture/hash manifest. Export parity passed. Initial safe checkpoint loading rejected NumPy string metadata; converted identifiers to built-in strings in our own locally generated checkpoint, verified all model tensor hashes unchanged, retained original checkpoint and audit. Future training emits native strings directly.
- SapiMouse ASGI replay:427 actual dev input windows,76 decisions×24claims. Baseline, selected cosine and author-default OCSVM scores match frozen artifacts exactly (maximum difference 0), with all threshold decisions/FAR/FRR matching. Test, remote and cross-origin access 403; malformed/oversized numeric inputs 422. No listener opened. Saved pointer_sapimouse_api_replay.json.
- Exposed the exported FCN/preprocessing contract to the keystroke agent for a genuinely paired BEACON experiment. No pointer/keystroke subjects are fabricated or matched across datasets here.
- Metric nuance: author-default OCSVM macro per-user EER is7.73%, slightly better than selected cosine's8.27%; cosine improves pooled EER/global threshold behavior. Report both definitions and do not imply every metric improved. Handcrafted baseline macro EER22.91%.
- Final router hardening verifies the exported encoder SHA-256 before loading it and hides SapiMouse until frozen results/export exist. Targeted Sapi ASGI replay passed again after that change, with exact score/decision agreement.

## 2026-09-20 — Runtime-only thread optimization and bot-rule coverage

- Device agent measured representative ExtraTrees latency with20 alternating generated-input repeats: median70.19ms at2threads versus17.85ms at1thread. Before changing runtime, pointer_runtime_thread_parity.py checked all9Balabit claims×278dev windows (2,502comparisons) at every frozen0.1%,1%,5%training-FAR threshold. Probabilities and decisions matched exactly (maximum difference0).
- Enabled n_jobs=1 only when loading Balabit inference estimators. Trained trees, artifact files, features and thresholds remain unchanged. Pointer dashboard explicitly labels pooled EER and states the author-versus-selected macro-EER qualification.
- Added frozen production bot-rule validation under research/benchmarks/bot_rules/. Evaluated5,100genuine strict CMU dev rows against each account's first10training-session1 priors:15false flags (0.294%), all impossible_hold, spread over12of51accounts. Constant weak no_probe was present because environment was intentionally empty. No fake pointer/browser context, no database and no listener.
- Separate fixed-seed synthetic training clones at0/2/5/10ms jitter were flagged100%/100%/90.20%/1.57%. These are stress tests, never real heldout bot FAR. Source trust fields are unknown and reconstructed true; this cannot validate isTrusted detection. Existing CMU-based rule history prevents a pristine-holdout claim. No production rule/threshold changed.
