# Four login combinations: implementation and synthetic screening

All four combinations are implemented in separate branches and worktrees. **No candidate is promoted to main.** The selection record was frozen before the corrected DEV replay: every candidate exceeded the 1% synthetic calibration FAR gate. These results compare implemented policies on artificial scenarios; they do not establish real-person app FAR/FRR.

## What runs

- **Control:** original statistical typing, device/bot checks and original step-up routing.
- **Typing step-up:** strict typing admission; uncertain cases request three extra typings. An RBF SVM is fitted from personal enrollment plus a compatible TRAIN background bank. Incompatible password schemas explicitly fall back to the original distance model.
- **Pointer step-up:** same typing challenger, with two target runs scored using the existing motor features on desktop; mobile uses cognitive keypad timing.
- **Keypad step-up:** same typing challenger, with two target runs scored on search timing and cadence.

The target widget is the existing scrambled keypad, so motor and cognitive branches receive identical recordings. This version does **not** integrate the SapiMouse neural encoder. All candidate branches revoke a prior login cookie while a new verification is pending. Bot/password failures remain blocking. Missing motor data never counts as a pass.

## Frozen dataset

There are **39 synthetic profiles / 390 scenarios**: 12 TRAIN, 12 calibration, 15 DEV. Each branch replays all 390 cases, totaling **1,560 complete policy scenarios** through the actual APIs. DEV contains **60 genuine / 90 impostor cases** per branch. Twelve additional, disjoint CMU identities supply background fitting and separate-session calibration examples. Model fitting never uses a probe.

Typing comes from real CMU recordings; target trajectories, search/hold timing, device attributes and event trust are simulated. No real people are linked across unrelated datasets. The ten scenario types include changed devices, mobile, cadence drift, same-device attacks, scripted inputs, typing-match attacks and keypad-match attacks. Equal scenario weights are a stress-test choice, not estimated attack prevalence. Enrollment assumes access to a physical keyboard; mobile-only signup is not implemented.

Shared cohort SHA-256: `8b1038a3a4d0a5140e5fbdb6a9db9387a11af771170697a32ceb17dd655ddecb`.
Shared background SHA-256: `290ca91fc9e7c68f950d1dd0e310f491c617d20047c13184a76495827c42d1fe`.

V1 failed before scoring because of an incorrect blank-cell constant. V2 mobile fixtures contained only one keypress for ten characters; they were correctly blocked. V3 fixes only virtual-key event counts; enrollment, target runs, splits and nonmobile cases remain bit-identical to v2. Earlier results are retained. This is a corrected exploratory evaluation with prior DEV exposure, not a pristine held-out claim. No model threshold was adjusted after DEV inspection.

## Corrected synthetic DEV results

| Branch | False accepts | Synthetic FAR | Genuine failures | Synthetic FRR | Genuine step-up rate |
|---|---:|---:|---:|---:|---:|
| login-control | 32/90 | 35.56% | 5/60 | 8.33% | 48.33% |
| typing-stepup | 10/90 | 11.11% | 25/60 | 41.67% | 58.33% |
| typing-pointer | 14/90 | 15.56% | 13/60 | 21.67% | 58.33% |
| typing-keypad | 13/90 | 14.44% | 16/60 | 26.67% | 58.33% |

FAR = final unauthorized allows / impostor scenarios. FRR = final nonallows / genuine scenarios after the prescribed step-up. Step-up rate is measured on genuine initial decisions. End-to-end policy EER is unavailable because the branching policy has no prespecified single scalar threshold sweep. Individual-model research EERs must not be substituted here.

| Scenario | Control accepts | Typing step-up accepts | Pointer step-up accepts | Keypad step-up accepts | Attempts per branch |
|---|---:|---:|---:|---:|---:|
| genuine_drift | 13 | 7 | 10 | 8 | 15 |
| genuine_mobile | 15 | 13 | 13 | 13 | 15 |
| genuine_new_device | 14 | 5 | 12 | 11 | 15 |
| genuine_same_device | 13 | 10 | 12 | 12 | 15 |
| impostor_mobile | 6 | 2 | 2 | 2 | 15 |
| impostor_new_device | 2 | 0 | 2 | 0 | 15 |
| impostor_same_device | 6 | 0 | 1 | 1 | 15 |
| keypad_match_attack | 6 | 0 | 1 | 1 | 15 |
| scripted_attack | 0 | 0 | 0 | 0 | 15 |
| typing_match_attack | 12 | 8 | 8 | 9 | 15 |

Typing-match attacks deliberately supply genuine owner typing with an impostor's target behavior. Their success exposes the limits of direct admission on typing alone. Mobile results depend on simulated keypad behavior and cannot validate real touch authentication.

## Verification and use

- Every branch consumes the same checksum-verified cohort, mappings, attempt IDs and split order.
- Actual registration, enrollment, login, challenge issuance, step-up and session-cookie outcomes were exercised in isolated temporary databases. No production database was read or copied, and no listener was started.
- Candidate policy/scorer checks: 71 passed on typing-stepup; 8 branch-policy checks passed on each other candidate. Control challenge/step-up/session checks: 36 passed, 1 pre-existing skip. Desktop and mobile browser smoke checks passed for all four branches with mocked APIs; those UI checks are separate from actual API replay.
- Elapsed server timing in result JSON includes password verification, cold fitting and concurrent benchmark load. It is not an isolated model latency or human task-duration measurement.

To try a branch manually, run `bash .worktrees/<branch-short-name>/experiments/run.sh` from the original workspace. Default localhost ports are 8004 (control), 8005 (typing-stepup), 8006 (typing-pointer), 8007 (typing-keypad). These scripts create separate branch databases and use bigidea. They have not been launched. See `README.md` for enrollment and compatibility limits.

The four implementations are ready for comparison. The next experiment should improve the TRAIN-side operating tradeoff and examine whether direct admission needs an additional signal; this DEV set must not be reused as untouched confirmation.
