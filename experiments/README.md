# Login combination experiment: typing-pointer

This worktree starts from `0e11268`. Normal login implements the policy described in `../code/bioprint/EXPERIMENT.md`. No change has been merged into main.

## Run locally

From this worktree run `bash experiments/run.sh`. The script uses bigidea, binds only 127.0.0.1 and assigns an isolated database under `experiments/runtime/`. It locates the original workspace through Git's common directory. It does not copy production users. The shared TRAIN typing background is checksum-verified. Default ports: control 8004, typing step-up 8005, pointer 8006, keypad 8007. Use separate browser profiles when comparing ports, because session cookies are hostname-scoped.

The source CMU password is `.tie5Roanl`; only an exact feature-schema match activates the background-trained SVM. Other passwords use the existing enrollment-distance model, explicitly identified in the response. Neither path has established real-user FAR for this application. Enrollment uses one warmup plus ten counted typings and six target runs. Mobile-only enrollment remains unsupported; mobile tests assume an existing account enrolled with a physical keyboard.

## Shared evaluation

All branches consume the exact same read-only `research/benchmarks/login_combinations_v3/cohort.json.gz` and `background.json` in the original workspace. `cohort_manifest.json` pins both. The cohort has 39 synthetic profiles / 390 scenarios: 12 TRAIN, 12 calibration, 15 DEV. Twelve additional CMU identities supply the SVM background bank, disjoint from all cohort identities. CMU sessions 7–8 are skipped before numeric decoding.

Typing traces come from CMU. Target search, motor traces, device context and event trust are simulated, not measurements linked across human datasets. Trust fields do not validate automation resistance. Controlled pointer step-up uses the existing keypad motor features; the SapiMouse neural model is not integrated in this version. A same-device typing-match attack intentionally substitutes the owner's typing while keeping the impostor's target behavior.

For replay, use the original workspace's `bash scripts/research.sh` wrapper and invoke this worktree's `experiments/replay.py --cohort <shared-directory> --split train|calibration|dev --output <new-report.json>`. It creates and deletes a temporary database, checks live challenge issuance and session cookies, and never opens a listener. It refuses to overwrite reports.

`browser_smoke.py` separately exercises desktop/mobile interaction with mocked API responses; it is not an end-to-end biometric test. The replay exercises actual API scoring. Failed early fixtures remain preserved in v1/v2. V1 used an incorrect blank-cell constant; v2 emitted one virtual-key press rather than ten. V3 changes only the virtual-key event count; all enrollment and nonmobile recordings remain identical to v2. Prior DEV exposure is disclosed; no threshold was changed based on it.

## Interpretation

Report final false accepts / impostor attempts, final genuine failures / genuine attempts, step-up frequency, signal coverage and latency. Scenario proportions are artificial, not attack prevalence. There is no prespecified scalar score for the branching policy, so policy EER is unavailable. No real-person app-wide accuracy claim or automatic main-branch promotion is justified by this simulation.
