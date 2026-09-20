"""Audit and report the frozen aligned experiment; never refit or select."""
import json
from pathlib import Path
import time
import numpy as np
import joblib
from audit import digest, check_counts
from infer import score_input

HERE = Path(__file__).parent
OUT = HERE / 'aligned_results'


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    for key, path in [('protocol_sha256', HERE / 'ALIGNED_PROTOCOL.md'), ('script_sha256', HERE / 'aligned.py'),
                      ('engine_sha256', HERE.parents[1] / 'code/bioprint/engine/aligned_typing.py')]:
        assert digest(path) == plan[key]
    frozen = json.loads((OUT / 'frozen.json').read_text())
    report = json.loads((OUT / 'report.json').read_text())
    control = json.loads((HERE / 'results/control.json').read_text())
    for item in frozen.values():
        assert digest(OUT / item['file']) == item['sha256']
        assert item['calibration']['far'] <= .01
    for key, row in report['results'].items():
        check_counts(row, OUT / f'{key.replace("/", "-")}-scores.npz')
        track = key.split('/')[0]
        assert not set(row['subjects']) & set(report['fit_profiles'][track])
        assert not set(row['subjects']) & set(report['calibration_profiles'][track])
        for field in ('genuine_attempts', 'impostor_attempts'):
            assert row[field] == control[track][field]
    sample = {'keystrokes': [e for i, code in enumerate(['KeyA', 'KeyX', 'Digit4', 'Period', 'KeyB'])
              for e in ({'code': code, 'type': 'down', 't': i * 130., 'field': 'password'},
                        {'code': code, 'type': 'up', 't': i * 130. + 80., 'field': 'password'})]}
    payload = {'enrollment': [sample] * 10, 'attempt': sample}
    latencies = {}
    for name in ('distance', 'logistic', 'extra_trees'):
        artifact = joblib.load(OUT / frozen[f'pooled/{name}']['file'])
        times = []
        score_input(payload, artifact)
        for _ in range(10):
            start = time.perf_counter()
            result = score_input(payload, artifact)
            times.append((time.perf_counter() - start) * 1000)
            assert np.isfinite(result['score'])
        latencies[name] = float(np.median(times))
    (OUT / 'audit.json').write_text(json.dumps({'verified_score_tables': len(report['results']),
        'verified_model_hashes': len(frozen), 'source_hashes_verified': True,
        'identity_disjointness_verified': True, 'same_trials_as_control': True,
        'generated_browser_input_median_ms': latencies}, indent=2))
    lines = ['# General matcher with personal password alignment', '',
        'This is the second, exploratory experiment on branch `experiment/general-typing`. '
        'It was declared after the summary-only model failed on DEV, so reused DEV cannot '
        'provide independent confirmation. No server or login policy was changed.', '',
        '## How it works for different passwords', '',
        'Each user still enrolls their own password ten times. The existing profile supplies '
        'a center c_j and spread s_j for each key-position timing. For a new attempt, compute '
        'z_j = clip((x_j-c_j)/s_j, -6, 6). Summarize these residuals separately for H, DD and '
        'UD timings: five absolute quantiles, absolute mean/standard deviation and signed '
        'median/mean/standard deviation. Append the existing distance/threshold ratio. '
        'These 31 features have the same meaning and size for different passwords. '
        'The shared logistic/tree model sees neither key labels nor account IDs. '
        'Alignment remains local to the account; unrelated passwords are never aligned.', '',
        'A new password therefore does not require retraining the shared model, but does '
        'require a new personal enrollment profile. This is not automatic online adaptation.', '',
        '## Evaluation', '',
        'Same guarded CMU and KeyRecs-fixed data, identity roles and trials as RESULTS.md. '
        'Fit, calibration and evaluation identities are disjoint. All nine model/threshold '
        'combinations were frozen before this follow-up read DEV. Thresholds target empirical '
        'calibration FAR<=1%; DEV FAR can differ. No hyperparameter search or threshold retuning. '
        'The distance comparator is also recalibrated, so gains cannot be credited merely to '
        'tightening the shipped threshold. Free text and gameplay are excluded here because '
        'they do not repeat the enrolled password sequence.', '',
        'FAR=FA/N_impostor, FRR=FR/N_genuine. EER is a diagnostic pooled empirical ROC statistic, '
        'not a deployable threshold. Comparisons share users and are dependent. '
        'This cohort has ten CMU and fourteen KeyRecs evaluation identities; these numbers '
        'must not be substituted into the earlier full-cohort reports.', '',
        '| Dataset | Training / method | FAR | FRR | EER | FA / impostors | FR / genuine | EER change vs shipped scorer (pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for track in ('cmu', 'fixed'):
        rows = [('Shipped positional scorer', control[track])]
        rows += [(key.removeprefix(track + '/'), r) for key, r in report['results'].items() if key.startswith(track + '/')]
        for name, r in rows:
            delta = 100 * (r['eer'] - control[track]['eer'])
            lines.append(f'| {track} | {name} | {100*r["far"]:.2f}% | {100*r["frr"]:.2f}% | {100*r["eer"]:.2f}% | {r["false_accepts"]}/{r["impostor_attempts"]} | {r["false_rejects"]}/{r["genuine_attempts"]} | {delta:+.2f} |')
    lines += ['', 'Negative EER change means improvement. `cmu_only` on KeyRecs and '
              '`keyrecs_only` on CMU are cross-dataset/password transfer. All other rows are '
              'unseen-user evaluation with the dataset represented in population fitting. '
              'Only two fixed password texts are represented, limiting generalization claims.', '',
              '## Verification and runtime', '',
              f'Fitting and evaluation took {report["seconds"]:.1f} seconds. The audit verified '
              '18 score tables, nine model hashes, source hashes, identity separation and '
              'identical trial counts against the shipped scorer. Fifteen tests cover the two '
              'representations and browser adapters. Median generated-input inference latency '
              '(including personal profile construction, excluding artifact load): ' +
              ', '.join(f'{k} {v:.2f} ms' for k, v in latencies.items()) + '.', '',
              '## Reproduce and use offline', '', '```bash',
              'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/aligned.py',
              'bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/finish_aligned.py',
              'bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/infer.py --representation aligned --model pooled/logistic --input recording.json',
              '```', '',
              'The run refuses an existing aligned_results directory. Keep original evidence. '
              'Input contains ten browser-format enrollment Sample objects and one attempt. '
              'The command returns a behavioral score, not authentication authorization. '
              'Credentials, bot checks and device context remain separate.', '',
              'All model binaries and score arrays are available locally; hashes and reports '
              'are committed. No candidate is deployed automatically. These are single-attempt '
              'typing results, not three-repetition, pointer-step-up or all-feature app error rates.', '']
    (HERE / 'ALIGNED_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'audit': 'passed', 'latency_ms': latencies}, indent=2))


if __name__ == '__main__':
    main()
