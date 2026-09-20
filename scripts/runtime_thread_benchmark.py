"""Generated-input inference timing and frozen-dev decision parity for RF threads.

No fitting, parameter selection on dev, artifact rewriting, or test reads.
Runtime recommendation is made from generated-input timing before dev validation.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'scripts'), str(ROOT / 'code/bioprint')]
import joblib
import numpy as np

OUT = ROOT / 'research/benchmarks/runtime_threads.json'


def timed_pair(models, predict, x, repeats=20):
    for model in models.values():
        for _ in range(3):
            predict(model, x)
    elapsed = {jobs: [] for jobs in models}
    for i in range(repeats):
        for jobs in ([1, 2] if i % 2 else [2, 1]):
            begin = time.perf_counter_ns()
            predict(models[jobs], x)
            elapsed[jobs].append((time.perf_counter_ns() - begin) / 1e6)
    return {str(jobs): {'median_ms': float(np.median(values)),
                        'p95_ms': float(np.quantile(values, .95)), 'repeats': repeats}
            for jobs, values in elapsed.items()}


def predict_in_batches(model, x):
    return np.concatenate([model.predict_proba(x[start:start+256])[:, 1]
                           for start in range(0, len(x), 256)])


def main():
    if OUT.exists():
        raise ValueError('Existing runtime report preserved; choose a new version before rerunning')
    report = {'timing_inputs': 'Generated finite vectors only, seeded uniform0..1',
              'timing_protocol': '3 warmups,20 alternating-order repeats, one query and256 query batch',
              'thread_environment': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
              'tabular': {}, 'representative_other_families': {}, 'test_read': False,
              'serialized_artifacts_changed': False}
    rng = np.random.default_rng(20260920)
    candidates = []
    # Phase one: all timing conclusions precede dev matrix access.
    for dataset in ['device', 'delbot', 'cognitive', 'touch_tsi']:
        directory = ROOT / 'research/benchmarks' / dataset
        schema = json.loads((directory / 'schema.json').read_text())
        for name, spec in schema['models'].items():
            path = directory / spec['artifact']
            model = joblib.load(path)
            if not hasattr(model, 'n_jobs') or not hasattr(model, 'estimators_'):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            models = {jobs: copy.deepcopy(model).set_params(n_jobs=jobs) for jobs in [1, 2]}
            generated = rng.uniform(0, 1, (256, schema['feature_count']))
            timings = {str(size): timed_pair(models, lambda m, x: m.predict_proba(x), generated[:size])
                       for size in [1, 256]}
            prefer_single = timings['1']['1']['median_ms'] < timings['1']['2']['median_ms'] * .8
            key = f'{dataset}/{name}'
            report['tabular'][key] = {'timings_by_batch_size': timings, 'singlethread_selected_from_generated_timing': prefer_single}
            candidates.append((key, directory, schema, name, path, digest, models))
    # Representative wrapper timing; owning agents perform full-family parity.
    for track in ['fixed', 'free']:
        path = ROOT / f'research/benchmarks/results/keyrecs-v1/{track}-train_selected.joblib'
        model = joblib.load(path)
        if getattr(model, 'kind', None) != 'extra_trees':
            continue
        models = {jobs: copy.deepcopy(model) for jobs in [1, 2]}
        for jobs, wrapper in models.items():
            wrapper.model.n_jobs = jobs
        generated = rng.uniform(0, 1, (1, model.model.n_features_in_))
        report['representative_other_families']['keyrecs_' + track] = timed_pair(models, lambda m, x: m.scores(x), generated)
    path = sorted((ROOT / 'research/benchmarks/pointer').glob('extra_1_user*.joblib'))[0]
    model = joblib.load(path)
    models = {jobs: copy.deepcopy(model).set_params(n_jobs=jobs) for jobs in [1, 2]}
    generated = rng.uniform(0, 1, (1, model.n_features_in_))
    report['representative_other_families']['balabit_one_claim'] = timed_pair(models, lambda m, x: m.predict_proba(x), generated)
    # Phase two: fixed-runtime dev parity only, no changes to method or threshold.
    for key, directory, schema, name, path, digest, models in candidates:
        with np.load(directory / 'dev_features.npz', allow_pickle=False) as data:
            x, y = data['X'], data['y']
        two, one = predict_in_batches(models[2], x), predict_in_batches(models[1], x)
        parity = {'rows': len(x), 'max_absolute_probability_error': float(np.max(abs(two-one))),
                  'scores_allclose_1e12': bool(np.allclose(one, two, atol=1e-12, rtol=1e-12)), 'operating_points': {}}
        for target, point in schema['models'][name]['operating_points'].items():
            threshold = point['threshold_selected_on_training_calibration']
            original, changed = two >= threshold, one >= threshold
            far, frr = float(changed[y == 0].mean()), float((~changed[y == 1]).mean())
            parity['operating_points'][target] = {
                'decision_flips': int(np.sum(original != changed)), 'far': far, 'frr': frr,
                'offline_far_frr_exact': far == point['dev']['far'] and frr == point['dev']['frr']}
        row = report['tabular'][key]
        row['full_dev_parity'] = parity
        row['runtime_change_safe'] = (row['singlethread_selected_from_generated_timing'] and
            parity['scores_allclose_1e12'] and all(p['decision_flips'] == 0 and p['offline_far_frr_exact']
                                                for p in parity['operating_points'].values()))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: {'one_query_ms': row['timings_by_batch_size']['1'],
                           'runtime_change_safe': row['runtime_change_safe'], 'parity': row['full_dev_parity']}
                      for key, row in report['tabular'].items()}, indent=2))


async def validate_runtime_api():
    from modalities_api_replay import replay
    destination = OUT.with_name('runtime_thread_api_validation.json')
    if destination.exists():
        raise ValueError('Preserving previous API validation report')
    checks = {}
    for dataset in ['device', 'delbot', 'cognitive', 'touch_tsi']:
        directory = ROOT / 'research/benchmarks' / dataset
        schema = json.loads((directory / 'schema.json').read_text())
        paths = [directory / spec['artifact'] for spec in schema['models'].values()]
        before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        report_path = directory / 'api_replay_singlethread.json'
        await replay(dataset, report_path)
        after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        assert before == after
        checks[dataset] = {'artifact_sha256_unchanged': before, 'replay_report': str(report_path.relative_to(ROOT))}
    destination.write_text(json.dumps({'runtime_change': 'Validated RF loaders set n_jobs=1 only in memory',
                                      'datasets': checks, 'serialized_artifacts_changed': False}, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-replay', action='store_true', help='Validate enabled runtime loader, preserving original reports')
    if parser.parse_args().api_replay:
        asyncio.run(validate_runtime_api())
    else:
        main()
