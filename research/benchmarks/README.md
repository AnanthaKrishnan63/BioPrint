# Reproducible behavioral verification research

Run commands from the repository root with `bash scripts/research.sh`. This
launcher uses the `bigidea` environment and repository-local research dependencies,
caches, and temporary directories. See `requirements-research.txt` for versions.
On a fresh checkout, first create `bigidea` using `code/typing/environment.yml`,
then run `bash scripts/setup_research.sh`. CPU PyTorch is installed separately
from its official CPU wheel index. Existing environments need no reinstall.
`BIGIDEA_PYTHON` can point to a nondefault location for that environment.

## Evidence and partition rules

Read `PROTOCOL.md`, `../../logs/WORKLOG.md`, and `../../logs/EXPERIMENTS.md` before
running experiments. Existing frozen artifacts must not be overwritten casually.
Train includes internal fitting, selection, and threshold calibration subsets.
Dev probes are validation only. Unseen-user enrollment support is a separate
training role; it cannot train the population model. Test recordings stay sealed.
Historical CMU exposure is disclosed; current readers reject test access.

Datasets include CMU, KeyRecs2023, Balabit, SapiMouse, FPStalker, DELBOT, BEACON,
Touch TSI, and repeated Stroop/Flanker tasks. Source notes and manifests record provenance,
licenses, partition membership, and transfer limitations. Browser linkage is
not personal identity, transcription is not spontaneous typing, and gameplay is
not login behavior. Never pair unrelated datasets by invented user identities.

## API verification without starting a server

```bash
bash scripts/research.sh scripts/api_dataset_replay.py --in-process
bash scripts/research.sh scripts/beacon_api_replay.py
bash scripts/research.sh scripts/modalities_api_replay.py device
bash scripts/research.sh scripts/modalities_api_replay.py delbot
bash scripts/research.sh scripts/modalities_api_replay.py cognitive
bash scripts/research.sh scripts/modalities_api_replay.py touch_tsi
bash scripts/research.sh scripts/beacon_neural_api_replay.py
bash scripts/research.sh scripts/beacon_type2branch_api_replay.py
bash scripts/research.sh scripts/bot_api_replay.py
bash scripts/research.sh scripts/cmu_account_api_replay.py
bash scripts/research.sh scripts/modalities_api_replay.py beacon_nonlinear
```

These scripts refuse to overwrite existing replay reports. Inspect those reports
first. The isolated research API serves public dev feature vectors and evaluates
frozen models. It rejects test access and non-loopback clients. In-process ASGI
replay opens no listening socket and does not import the production database.
BEACON exercises actual paired keyboard/mouse/context inference; its small
gameplay cohort does not validate keypad, bots, or a complete login deployment.

`beacon_type2branch_dev_evaluation_v1/report.json` adds eight frozen paired
comparators on three DEV people: 243 claims, 140 windows, including 81 windows
shorter than the source model's 25-event training minimum. The Type2Branch
hybrid has FAR24.69%, FRR67.90%, EER40.74%, versus45.68%,30.86%,38.27% for the
matched old hybrid under the same TRAIN selection rule. Lower FAR comes with
worse FRR and EER. Prior DEV exposure remains disclosed. Actual in-process API
replay reproduced all eight models' 243 scores each exactly, with zero decision
changes at all three thresholds and exact pooled/per-person metrics. See
`beacon_type2branch_api_replay_v1/report.json`. The boundary is fusion scoring
on features from offline encoders; it does not establish live browser-to-encoder
verification. No listener was started.

## Focused verification

```bash
bash scripts/research.sh -m pytest -q code/bioprint/tests/test_strict_protocol.py code/bioprint/tests/test_research_modalities.py --basetemp=.research-tmp/pytest-research
```

Research backends remain opt-in. Production users require suitable enrollment,
aligned feature schemas, and calibrated thresholds before any model migration.
Report actual dev FAR alongside FRR at training-calibrated operating points;
diagnostic EER is not a deployable threshold. Retain negative experiments.

Run `bash scripts/research.sh scripts/dataset_budget.py` for a metadata-only disk
budget check. Snapshot `dataset_budget_hmog_replay.json` includes raw
data, derived arrays and conservatively counted temporary fixtures totaling
1,170,379,121 bytes (1.17 GB). The largest source including all temporary fixtures
is467,121,700 bytes; adding omitted HMOG source documents raises that conservative
bound to467,357,056 bytes, still below500MB. See `dataset_budget_hmog_replay_scope.json`
for accounting limits. Model checkpoints and optional Python dependencies
are accounted separately from dataset storage.

`summary_v11.json` and `summary_v11.csv` collect 72 DEV model rows from 25 frozen
source reports, with three infeasible experiments recorded separately,
including negative results and EER aggregation labels. Regenerate only after
reviewing the logs with `scripts/build_research_summary.py --output-stem <new-name>`; existing summaries
are protected against overwrite. The JSON records hashes of every source report.
`scripts/inference_latency.py` measures warm compute cost separately from data
collection; its report is not an end-to-end login latency guarantee.

The three-level arithmetic experiment is recorded in
`arithmetic_train_v1/README.md`. Its frozen DEV enrollment failed, so its TRAIN
rates are not added as DEV metric rows. The isolated research API exposes
`GET /api/arithmetic/results`; DEV samples and scoring return409 until a valid
complete-cohort experiment exists. Test requests return403. Run
`bash scripts/research.sh scripts/arithmetic_api_replay.py` only for a new,
deliberately prepared replay directory; the existing run is preserved in
`arithmetic_api_v1/replay.json`. This checks status/report fidelity, not biometric
score parity. No listener is needed for that replay.

Four BEACON wrapper rows retain the original logistic regression.
Their interpolated EER convention differs from the original report; unchanged
operating decisions and a smaller printed EER do not constitute an improvement.
The original `summary.json` and `summary.csv` remain available.

`cmu-account-selection/` records a fixed-grid per-account SVM follow-up.
At the training 1% FAR target, dev FAR is 0.949% and FRR 67.431%, versus
0.998% and 72.980% for the original global SVM. This is exploratory follow-up
on previously exposed data, not independent confirmation or near-zero error.

`cmu-far-selection/` instead selects each model for training FRR at 1% FAR.
Dev FRR falls to 64.10% at actual FAR 0.99%, but EER worsens to 18.50%.
This is an operating-point tradeoff, not a universally better replacement.
Replay all four frozen controls with
`bash scripts/research.sh scripts/cmu_account_api_replay.py --dataset cmu-far-selection`.

A 12-subject HMOG subset adds repeated paired mobile typing/touch/motion
archives (about 428.5 MB including discovery metadata). Identity roles were
frozen before acquisition; reserved identities were not fetched. Session roles
are frozen. After preserving zero-window strict extraction and failed original
calibration, a separately declared retained-pair/seen-identity-calibration variant
was trained and validated. Full-cohort DEV status remains infeasible: two accounts
have no eligible probes, and31 observed probes cover20.13% of154 inspected candidate
windows. Joint EER45.16%,FAR0%/FRR100% at the1% TRAIN target are poor conditional
results. An explicit partial API release preserves this status and missing coverage;
372 observed scores match offline decisions exactly. No test observations were
read, no listener started and no production model promoted. See `hmog/README.md`.
