"""Verify and summarize all four policies on the shared frozen synthetic cohort."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

root = Path(os.environ['BIOPRINT_RESEARCH_ROOT'])
base = root/'research/benchmarks/login_combinations_v3'
names = ['login-control','typing-stepup','typing-pointer','typing-keypad']
manifest = json.loads((base/'manifest.json').read_text())
reports = {}
rows = []
for split in ['train','calibration','dev']:
    expected_ids = None
    for name in names:
        path = base/f'{name}-{split}-v1.json'
        d = json.loads(path.read_text())
        assert d['manifest']['files'] == manifest['files']
        ids = [a['id'] for a in d['attempts']]
        if expected_ids is None: expected_ids=ids
        assert ids == expected_ids
        g=[a for a in d['attempts'] if a['genuine']]
        i=[a for a in d['attempts'] if not a['genuine']]
        m=d['overall']
        assert m['false_accepts']==sum(a['final']=='allow' for a in i)
        assert m['false_rejects']==sum(a['final']!='allow' for a in g)
        assert m['far']==m['false_accepts']/len(i)
        assert m['frr']==m['false_rejects']/len(g)
        assert m['genuine_stepup_n']==sum(a['initial'] in ('step_up','keypad') for a in g)
        rows.append(dict(split=split,mode=name,**m,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        reports[(split,name)]=d

selection=json.loads((base/'selection_before_dev.json').read_text())
table='| Branch | False accepts | Synthetic FAR | Genuine failures | Synthetic FRR | Genuine step-up rate |\n|---|---:|---:|---:|---:|---:|\n'
for name in names:
    d=reports[('dev',name)]['overall']
    table+=f"| {name} | {d['false_accepts']}/{d['impostor_n']} | {100*d['far']:.2f}% | {d['false_rejects']}/{d['genuine_n']} | {100*d['frr']:.2f}% | {100*d['genuine_stepup_n']/d['genuine_n']:.2f}% |\n"
scenarios='| Scenario | Control accepts | Typing step-up accepts | Pointer step-up accepts | Keypad step-up accepts | Attempts per branch |\n|---|---:|---:|---:|---:|---:|\n'
for scenario in sorted(reports[('dev',names[0])]['by_scenario']):
    counts=[]
    for name in names:
        a=[a for a in reports[('dev',name)]['attempts'] if a['scenario']==scenario]
        counts.append(str(sum(x['final']=='allow' for x in a)))
    scenarios+=f"| {scenario} | {' | '.join(counts)} | {len(a)} |\n"
report=f'''# Four login combinations: implementation and synthetic screening

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

Shared cohort SHA-256: `{manifest['files']['cohort.json.gz']}`.
Shared background SHA-256: `{manifest['files']['background.json']}`.

V1 failed before scoring because of an incorrect blank-cell constant. V2 mobile fixtures contained only one keypress for ten characters; they were correctly blocked. V3 fixes only virtual-key event counts; enrollment, target runs, splits and nonmobile cases remain bit-identical to v2. Earlier results are retained. This is a corrected exploratory evaluation with prior DEV exposure, not a pristine held-out claim. No model threshold was adjusted after DEV inspection.

## Corrected synthetic DEV results

{table}
FAR = final unauthorized allows / impostor scenarios. FRR = final nonallows / genuine scenarios after the prescribed step-up. Step-up rate is measured on genuine initial decisions. End-to-end policy EER is unavailable because the branching policy has no prespecified single scalar threshold sweep. Individual-model research EERs must not be substituted here.

{scenarios}
Typing-match attacks deliberately supply genuine owner typing with an impostor's target behavior. Their success exposes the limits of direct admission on typing alone. Mobile results depend on simulated keypad behavior and cannot validate real touch authentication.

## Verification and use

- Every branch consumes the same checksum-verified cohort, mappings, attempt IDs and split order.
- Actual registration, enrollment, login, challenge issuance, step-up and session-cookie outcomes were exercised in isolated temporary databases. No production database was read or copied, and no listener was started.
- Candidate policy/scorer checks: 71 passed on typing-stepup; 8 branch-policy checks passed on each other candidate. Control challenge/step-up/session checks: 36 passed, 1 pre-existing skip. Desktop and mobile browser smoke checks passed for all four branches with mocked APIs; those UI checks are separate from actual API replay.
- Elapsed server timing in result JSON includes password verification, cold fitting and concurrent benchmark load. It is not an isolated model latency or human task-duration measurement.

To try a branch manually, run `bash .worktrees/<branch-short-name>/experiments/run.sh` from the original workspace. Default localhost ports are 8004 (control), 8005 (typing-stepup), 8006 (typing-pointer), 8007 (typing-keypad). These scripts create separate branch databases and use bigidea. They have not been launched. See `README.md` for enrollment and compatibility limits.

The four implementations are ready for comparison. The next experiment should improve the TRAIN-side operating tradeoff and examine whether direct admission needs an additional signal; this DEV set must not be reused as untouched confirmation.
'''
comparison=dict(rows=rows,selection=selection,manifest=manifest,synthetic_only=True)
(base/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
(base/'RESULTS.md').write_text(report)
for name in names:
    dest=root/'.worktrees'/name/'experiments'
    (dest/'results').mkdir(exist_ok=True)
    for path in [base/'comparison.json',base/'selection_before_dev.json',*base.glob('*-v1.json')]:
        shutil.copyfile(path,dest/'results'/path.name)
    shutil.copyfile(base/'RESULTS.md',dest/'RESULTS.md')
    shutil.copyfile(root/'research/benchmarks/login_combinations_v2/browser-smoke.json',dest/'results/browser-smoke.json')
print(table)
print('Verified identical inputs/order and recomputed all 12 result summaries.')
