"""One frozen CMU account-selection dev comparison; no fitting or tuning."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import numpy as np
from threadpoolctl import threadpool_limits
from eval import strict_cmu as cmu

OUT = ROOT / 'research/benchmarks/cmu-account-selection'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    paths = [OUT / name for name in ['dev_results.json', 'dev_scores.npz', 'validation_plan.json']]
    if any(p.exists() for p in paths):
        raise SystemExit('Preserve prior validation; do not overwrite or repeat')
    artifact_path = OUT / 'frozen_models.json'
    artifact = json.loads(artifact_path.read_text())
    assert digest(Path(cmu.__file__)) == artifact['protocol']['source_sha256']
    original = ROOT / 'research/benchmarks/cmu/frozen_models.json'
    assert digest(original) == artifact['protocol']['original_artifact_sha256']
    plan = {'artifact_sha256': digest(artifact_path), 'validator_sha256': digest(__file__),
            'strict_loader_sha256': digest(Path(cmu.__file__)), 'dev_sessions': [5, 6],
            'models': list(artifact['models']), 'threshold_source': 'frozen session4 calibration',
            'paired_bootstrap': {'resamples': 2000, 'seed': 20260920, 'unit': 'account', 'interval': [.025, .975]},
            'test_measurements_parsed': False,
            'caveat': 'Legacy and prior global-dev exposure; exploratory follow-up. Account bootstrap does not fully model dependence from shared impostor probes.'}
    cmu.write_json(paths[2], plan)  # Must precede loading new development measurements.
    started = time.monotonic()
    names, records = cmu.load_partition('dev')
    subjects = sorted(records)
    x = np.concatenate([records[s]['X'] for s in subjects])
    y = np.concatenate([np.full(len(records[s]['X']), j, dtype=int) for j, s in enumerate(subjects)])
    rng = np.random.default_rng(20260920)
    indices = rng.integers(0, len(subjects), size=(2000, len(subjects)))
    result = {'protocol': artifact['protocol'], 'artifact_sha256': digest(artifact_path),
              'selected_on_training': 'account_selected', 'split': 'validation_dev', 'results': {},
              'validation_plan': plan, 'paired_account_deltas': {}}
    arrays = {'y': y, 'subjects': np.array(subjects), 'feature_names': np.array(names)}
    with threadpool_limits(limits=2):
        for method, models in artifact['models'].items():
            assert list(sorted(models)) == subjects
            matrix = np.column_stack([cmu.distances(models[s], x) for s in subjects])
            arrays[method] = matrix
            per_user = {}
            for j, subject in enumerate(subjects):
                genuine, impostor = matrix[y == j, j], matrix[y != j, j]
                model = models[subject]
                per_user[subject] = {'eer_diagnostic': cmu.eer(genuine, impostor),
                    'genuine_count': len(genuine), 'impostor_count': len(impostor),
                    'operating_points': {key: cmu.rates(genuine, impostor, threshold)
                                         for key, threshold in model['calibration'].items()}}
                if method == 'baseline':
                    per_user[subject]['shipped_threshold'] = cmu.rates(genuine, impostor, model['threshold'])
            values = np.array([per_user[s]['eer_diagnostic'] for s in subjects])
            summary = {'macro_eer_diagnostic': float(values.mean()),
                       'subject_bootstrap_95pct_ci': np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
                       'operating_points': {key: {rate: float(np.mean([per_user[s]['operating_points'][key][rate] for s in subjects]))
                                                for rate in ['far', 'frr']} for key in ['eer', 'far_1pct', 'far_5pct']}}
            if method == 'baseline':
                summary['shipped_threshold'] = {rate: float(np.mean([per_user[s]['shipped_threshold'][rate] for s in subjects])) for rate in ['far', 'frr']}
            result['results'][method] = {'summary': summary, 'per_user': per_user}
    chosen = result['results']['account_selected']['per_user']
    for reference in ['baseline', artifact['protocol']['original_selected']]:
        control = result['results'][reference]['per_user']
        differences = {'eer_diagnostic': np.array([chosen[s]['eer_diagnostic'] - control[s]['eer_diagnostic'] for s in subjects])}
        for point in ['far_1pct', 'far_5pct']:
            for rate in ['frr', 'far']:
                differences[f'{rate}_at_training_{point}'] = np.array([
                    chosen[s]['operating_points'][point][rate] - control[s]['operating_points'][point][rate] for s in subjects])
        result['paired_account_deltas'][f'account_selected_minus_{reference}'] = {
            metric: {'mean_delta': float(values.mean()),
                     'paired_account_bootstrap_95pct_ci': np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
            for metric, values in differences.items()}
    result['elapsed_seconds'] = time.monotonic() - started
    result['score_artifact_direction'] = 'larger_is_impostor; research API negates to larger_is_genuine'
    result['test_measurements_parsed'] = False
    np.savez_compressed(paths[1], **arrays)
    result['dev_scores_sha256'] = digest(paths[1])
    cmu.write_json(paths[0], result)
    (OUT / 'validation_source.py').write_text(Path(__file__).read_text())
    print(json.dumps({'results': {k: v['summary'] for k, v in result['results'].items()},
                      'paired_account_deltas': result['paired_account_deltas'],
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    validate()
