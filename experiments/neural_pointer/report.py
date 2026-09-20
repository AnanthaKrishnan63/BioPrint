"""Fixed-threshold tradeoff and pairing sensitivity; no fitting or selection."""
import copy
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['BIOPRINT_RESEARCH_ROOT'])
BRANCH = Path(__file__).resolve().parents[2]
RESULTS = BRANCH / 'experiments/results'
report = json.loads((RESULTS / 'typing-pointer-neural-dev-v2.json').read_text())
frozen = json.loads((ROOT / 'research/benchmarks/pointer_sapimouse_optimized/frozen_training_selection.json').read_text())
thresholds = frozen['thresholds']['cosine@5']


def stats(rows):
    genuine = [r for r in rows if r['genuine']]
    impostor = [r for r in rows if not r['genuine']]
    fa = sum(r['final'] == 'allow' for r in impostor)
    fr = sum(r['final'] != 'allow' for r in genuine)
    return {'false_accepts': fa, 'impostor_n': len(impostor), 'false_rejects': fr, 'genuine_n': len(genuine),
            'far': fa / len(impostor), 'frr': fr / len(genuine)}


curve = {}
for target, threshold in thresholds.items():
    rows = copy.deepcopy(report['results']['neural'])
    for row in rows:
        if row['neural_used'] and row['pointer_similarity'] is not None:
            row['final'] = 'allow' if row['pointer_similarity'] >= threshold else 'block'
    curve[target] = {'threshold': threshold, **stats(rows)}
assert curve['0.01']['false_accepts'] == report['overall']['neural']['false_accepts']
assert curve['0.01']['false_rejects'] == report['overall']['neural']['false_rejects']
with np.load(ROOT / 'research/benchmarks/pointer_sapimouse_optimized/cosine@5_dev_scores.npz', allow_pickle=False) as a:
    scores, owners, actual = a['scores'], a['users'], a['true_user']
identities = sorted(report['plan']['mapping'])
sensitivity = []
for seed in range(20260920, 20261020):
    rng = np.random.default_rng(seed)
    mapping = dict(zip(identities, rng.permutation(owners)[:len(identities)]))
    following = dict(zip(identities, list(mapping.values())[1:] + list(mapping.values())[:1]))
    rows = copy.deepcopy(report['results']['neural'])
    for row in rows:
        if not row['neural_used']:
            continue
        owner = row['id'].split('-')[0]
        source = mapping[owner] if row['genuine'] or row['scenario'] == 'keypad_match_attack' else following[owner]
        column = int(rng.choice(np.flatnonzero(actual == source)))
        similarity = float(scores[list(owners).index(mapping[owner]), column])
        row['final'] = 'allow' if similarity >= report['threshold'] else 'block'
    sensitivity.append({'seed': seed, **stats(rows)})
summary = {key: {'minimum': min(r[key] for r in sensitivity), 'median': float(np.median([r[key] for r in sensitivity])),
                 'maximum': max(r[key] for r in sensitivity)} for key in ('far', 'frr')}
result = {'threshold_tradeoff': curve, 'pairing_sensitivity': sensitivity, 'pairing_range': summary,
          'interpretation': 'Sensitivity to artificial pairing, not a real-world confidence interval; secondary score-level simulation assumes valid captures'}
(RESULTS / 'typing-pointer-neural-tradeoff.json').write_text(json.dumps(result, indent=2))
old, new = report['overall']['legacy'], report['overall']['neural']
lines = ['# Trained pointer integration: synthetic FAR / FRR', '',
    'The trained SapiMouse FCN + cosine model is now integrated into the desktop '
    '`typing-pointer` step-up path after personal pointer enrollment. This report '
    'measures a synthetic cross-dataset composition through the actual API and encoder, '
    'not real people using the combined app.', '',
    '## Paired comparison', '',
    '| Configuration | False accepts | FAR | False rejects | FRR |',
    '|---|---:|---:|---:|---:|']
for label, row in [('Previous pointer motor scorer', old), ('Trained pointer, current threshold', new)]:
    lines.append(f'| {label} | {row["false_accepts"]}/{row["impostor_n"]} | {row["far"]*100:.2f}% | {row["false_rejects"]}/{row["genuine_n"]} | {row["frr"]*100:.2f}% |')
lines += ['', f'FAR improves by {(old["far"]-new["far"])*100:.2f} percentage points; '
          f'FRR worsens by {(new["frr"]-old["frr"])*100:.2f} points. '
          'This is five fewer false accepts and six additional false rejects, not an across-the-board improvement.', '',
          f'Both policies require additional verification in {new["challenge_n"]}/150 cases. '
          f'The new encoder decides {new["neural_n"]}/150 desktop cases; mobile retains keypad verification. '
          f'Incomplete checks: {new["incomplete_n"]}. The replay reproduced every stored legacy decision '
          'and checked session-cookie outcomes after all 300 paired cases. All 33 neural cosine scores '
          'matched their frozen research predictions within 2e-6.', '',
          '## FAR versus FRR at existing TRAIN-calibrated thresholds', '',
          'These are the three thresholds already frozen by the pointer research. Only the 1% '
          'source-calibration target is currently used in enrollment. Other rows are diagnostic '
          'score-level counterfactuals on the same recorded attempts; no threshold was retuned or promoted.', '',
          '| Source calibration FAR target | Cosine threshold | App synthetic FA / impostors | FAR | FR / genuine | FRR |',
          '|---|---:|---:|---:|---:|---:|']
for target, row in curve.items():
    lines.append(f'| {float(target)*100:g}% | {row["threshold"]:.9f} | {row["false_accepts"]}/{row["impostor_n"]} | {row["far"]*100:.2f}% | {row["false_rejects"]}/{row["genuine_n"]} | {row["frr"]*100:.2f}% |')
lines += ['', 'The source calibration target is not the combined app FAR. The branching policy '
          'has no prespecified single scalar score, so an all-feature EER is not reported.', '',
          '## Why false accepts remain', '',
          'All nine remaining false accepts bypass the neural challenge: seven typing-match '
          'attacks pass direct admission, and two mobile impostor cases pass the unchanged keypad. '
          'Changing a desktop step-up scorer cannot catch an attempt that never reaches it. '
          'The next policy question is whether direct admission needs another independent signal, '
          'balanced against extra friction. This report does not change that policy.', '',
          '## Synthetic construction and limits', '',
          'The original 15 synthetic accounts and 150 DEV cases are unchanged. Their CMU typing '
          'is combined with separately enrolled SapiMouse users by a frozen seeded mapping. '
          'Real people are not linked across datasets. The run preserves typing/device/bot inputs '
          'and mobile keypad traces; only desktop challenge scoring changes. Typing in this '
          'cohort matches the CMU SVM schema. These results do not establish performance for '
          'an arbitrary-password account using the typing fallback.', '',
          'Each public pointer center comes from that source identity’s separate three-minute '
          'TRAIN-support recording. DEV capture windows contain 641 consecutive original '
          'coordinates, never padded or joined across sessions. The endpoint uses the actual '
          'hash-pinned encoder and first five blocks. The simulator advances the challenge clock '
          'only in its disposable database; real-server clock checks remain active.', '',
          'Across 100 additional deterministic artificial pairings, FAR ranges '
          f'{summary["far"]["minimum"]*100:.2f}–{summary["far"]["maximum"]*100:.2f}% '
          f'(median {summary["far"]["median"]*100:.2f}%), and FRR ranges '
          f'{summary["frr"]["minimum"]*100:.2f}–{summary["frr"]["maximum"]*100:.2f}% '
          f'(median {summary["frr"]["median"]*100:.2f}%). '
          'This measures dependence on pairing, not a confidence interval or new participants. '
          'These secondary simulations assume usable captures and do not rerun the API.', '',
          'V1 is retained: a mismatched adapter guard rejected stationary points in 19 recordings. '
          'V2 removes that invented guard while retaining the model, threshold, mapping and source '
          'coordinates. No biometric threshold tuning occurred; earlier exposure remains disclosed.', '',
          '## Enrollment and use', '',
          'Every desktop account without a neural pointer profile is guided to enrollment after '
          'a successful login. Three one-minute mouse/trackpad recordings are saved independently '
          'and form a per-account mouse profile. Existing typing and keypad profiles are preserved. '
          'The profile activates neural scoring only when desktop step-up is needed. Mobile uses '
          'the existing keypad. No user’s real pointer enrollment was fabricated.', '',
          'Restart the pointer branch manually with `bash .worktrees/typing-pointer/experiments/run.sh` '
          'from the original repository root, then open `http://localhost:8006/pointer-enroll.html` '
          'after signing in. The script loads the checksum-pinned encoder from the original workspace. '
          'No server was started or restarted by this experiment.', '',
          '## Reproduce', '', '```bash',
          'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/typing-pointer/experiments/neural_pointer/replay.py --version v2',
          'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/typing-pointer/experiments/neural_pointer/report.py',
          '```', '',
          'The replay refuses to overwrite a completed result. Source mappings, hashes, exact '
          'case outcomes and per-scenario rates are in `experiments/results/typing-pointer-neural-dev-v2.json`; '
          'all threshold and pairing results are in `typing-pointer-neural-tradeoff.json`.', '']
(BRANCH / 'experiments/NEURAL_POINTER_RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({'curve': curve, 'pairing_range': summary}, indent=2))
