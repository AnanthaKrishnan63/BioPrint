"""Measure frozen-model compute cost; never train or choose a model from timing."""
import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from research_api import samples, fitted_model
from research_modalities import artifact_directory, tabular_schema, tabular_model

ROOT = Path(__file__).resolve().parents[1]


def measure(function, repeats):
    async def run():
        async def invoke():
            result = function()
            return await result if inspect.isawaitable(result) else result
        for _ in range(3):
            await invoke()
        durations = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            await invoke()
            durations.append((time.perf_counter_ns() - start) / 1e6)
        return durations
    durations = asyncio.run(run())
    return {'median_ms': float(np.median(durations)),
            'p95_ms': float(np.quantile(durations, .95)),
            'min_ms': min(durations), 'repeats': repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/benchmarks/inference_latency.json')
    args = parser.parse_args()
    if args.repeats < 5:
        raise SystemExit('At least five repetitions required')
    if args.output.exists():
        raise SystemExit('Latency report exists; inspect logs before repeating')
    results = {}
    for dataset in ['cmu', 'keyrecs-fixed', 'keyrecs-free', 'keyrecs-typenet']:
        x, _ = samples(dataset, 'dev')
        model = fitted_model(dataset, 'train_selected')
        probe = x[:1]
        shape = model.scores(probe).shape
        results[dataset] = {'input': 'one dev query against every enrolled claim',
                            'claims_per_call': shape[1],
                            **measure(lambda: model.scores(probe), args.repeats)}
    for dataset in ['device', 'delbot', 'cognitive', 'touch_tsi']:
        with np.load(artifact_directory(dataset) / 'dev_features.npz', allow_pickle=False) as data:
            probe = data['X'][:1]
        for name in tabular_schema(dataset)['models']:
            model = tabular_model(dataset, name)
            results[f'{dataset}/{name}'] = {
                'input': 'one paired feature vector or one trajectory',
                'claims_per_call': 1,
                **measure(lambda: model.predict_proba(probe), args.repeats)}
    from research_pointer_api import spec, samples as pointer_samples, score, ScoreRequest
    for dataset in ['balabit', 'sapimouse']:
        config = spec(dataset)
        hand, sequence, _, groups = pointer_samples(dataset, 'dev')
        n = config['blocks']
        if len(set(groups[:n])) != 1:
            raise ValueError('Timing input must stay within one real recording')
        hand = hand[:n].astype(object)
        hand[~np.isfinite(hand.astype(float))] = None
        request = ScoreRequest(model='selected', features=hand.tolist(),
                               sequences=None if sequence is None else sequence[:n].tolist(),
                               session_ids=groups[:n].tolist())
        results[f'pointer/{dataset}'] = {
            'input': f'{n} contiguous blocks against every enrolled claim',
            'claims_per_call': len(config['subjects']),
            **measure(lambda: score(dataset, request), args.repeats)}
    report = {'python': platform.python_version(),
              'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
              'measurements': results,
              'limitations': ['Compute only: excludes capture time, feature extraction, network, HTTP serialization and cold loading.',
                              'Single-query all-account typing is more work than one claimed-account verification.',
                              'Wall time reflects concurrent host workload; not a service latency guarantee.',
                              'No threshold, feature, or model selection uses these measurements.'],
              'sealed_data_read': False}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
