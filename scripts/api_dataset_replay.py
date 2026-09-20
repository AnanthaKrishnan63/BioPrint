"""Replay public dev samples through the isolated research HTTP API.

Starts no server. Pass --base-url http://127.0.0.1:8011 to an explicitly launched
loopback research_api service. --in-process runs the identical ASGI endpoints.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import httpx
import numpy as np
from keystroke_benchmark import evaluate


async def replay(base_url, in_process, output, requested=None):
    transport = None
    if in_process:
        from research_api import app
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    report = {'mode': 'ASGI endpoint replay' if in_process else 'HTTP loopback endpoint replay',
              'test_accessed': False, 'datasets': {}}
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=60) as client:
        listing = await client.get('/api/datasets')
        listing.raise_for_status()
        available = listing.json()['datasets']
        if requested and not set(requested).issubset(available):
            raise ValueError('Requested datasets are not ready')
        for dataset in (requested or available):
            # Guard verification never returns sealed values.
            denied = await client.get(f'/api/datasets/{dataset}/test/samples')
            assert denied.status_code == 403
            offset, scores, labels, accepted, subjects, thresholds = 0, [], [], [], None, None
            start = time.perf_counter()
            while True:
                response = await client.get(f'/api/datasets/{dataset}/dev/samples', params={'offset': offset, 'limit': 128})
                response.raise_for_status()
                page = response.json()
                if not page['x']:
                    break
                scored = await client.post(f'/api/models/{dataset}/score', json={'features': page['x']})
                scored.raise_for_status()
                payload = scored.json()
                if subjects is not None:
                    assert payload['subjects'] == subjects
                    assert payload['thresholds'] == thresholds
                subjects, thresholds = payload['subjects'], payload['thresholds']
                scores.extend(payload['scores'])
                accepted.extend(payload['accepted'])
                labels.extend(page['y'])
                offset += len(page['x'])
                if offset >= page['total']:
                    break
            a, y = np.asarray(scores), np.asarray(labels)
            assert np.array_equal(np.asarray(accepted), a >= np.asarray(thresholds))
            if dataset == 'keyrecs-typenet':
                with np.load(ROOT / 'research/benchmarks/results/typenet-keyrecs-v1/dev-scores.npz', allow_pickle=False) as offline:
                    assert np.array_equal(y, offline['y'])
                    assert np.array_equal(np.asarray(subjects), offline['subjects'])
                    error = float(np.max(np.abs(a - offline['scores'])))
                    assert np.allclose(a, offline['scores'], rtol=0, atol=1e-5)
                    assert np.array_equal(np.asarray(thresholds), offline['thresholds'])
                    assert np.array_equal(np.asarray(accepted), offline['scores'] >= offline['thresholds'])
            elif dataset in {'cmu', 'cmu-account-selection', 'cmu-far-selection'}:
                from eval.strict_cmu import load_partition, distances
                artifact = json.loads((ROOT / 'research/benchmarks' / dataset / 'frozen_models.json').read_text())
                models = artifact['models'][artifact['selected']]
                names, records = load_partition('dev')
                assert subjects == sorted(models)
                x = np.concatenate([records[s]['X'] for s in subjects])
                offline_scores = -np.column_stack([distances(models[s], x) for s in subjects])
                error = float(np.max(np.abs(a - offline_scores)))
                assert np.allclose(a, offline_scores, rtol=0, atol=1e-10)
                assert thresholds == [-models[s]['calibration']['far_1pct'] for s in subjects]
                assert np.array_equal(np.asarray(accepted), offline_scores >= np.asarray(thresholds))
            else:
                track = dataset.removeprefix('keyrecs-')
                with np.load(ROOT / f'research/benchmarks/results/keyrecs-v1/{track}-train_selected-dev-scores.npz', allow_pickle=False) as offline:
                    assert np.array_equal(y, offline['y'])
                    assert np.array_equal(np.asarray(subjects), offline['subjects'])
                    error = float(np.max(np.abs(a - offline['scores'])))
                    assert np.allclose(a, offline['scores'], rtol=0, atol=1e-12)
                    assert np.array_equal(np.asarray(thresholds), offline['thresholds'])
                    assert np.array_equal(np.asarray(accepted), offline['scores'] >= offline['thresholds'])
            result = evaluate(a, y, subjects, np.asarray(thresholds))
            report['datasets'][dataset] = {'samples_replayed': len(y), 'max_absolute_score_error': error,
                'elapsed_seconds': time.perf_counter() - start,
                'test_guard_status': denied.status_code,
                'all_frozen_decisions_match': True,
                'metrics': result['aggregate_macro_user']}
            print(json.dumps({'dataset': dataset, **report['datasets'][dataset]}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as dst:
        json.dump(report, dst, indent=2)
        dst.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8011')
    parser.add_argument('--in-process', action='store_true')
    parser.add_argument('--datasets', nargs='+', help='Limit replay to newly changed model contracts')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/benchmarks/results/keyrecs-v1/api-replay.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; preserve previous replay results')
    asyncio.run(replay(args.base_url, args.in_process, args.output, args.datasets))


if __name__ == '__main__':
    main()
