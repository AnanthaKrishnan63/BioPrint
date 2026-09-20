"""Verify every paired neural BEACON development decision via isolated ASGI."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import httpx
import joblib
import numpy as np
from beacon_benchmark import measure


async def run(version):
    from research_api import app
    from research_beacon_neural_api import router
    if not any(getattr(r, 'path', '').startswith('/api/neural-beacon') for r in app.routes):
        app.include_router(router)
    base = ROOT / f'research/benchmarks/beacon-neural-{version}'
    frozen = json.loads((base / 'frozen.json').read_text())
    stored = json.loads((base / 'dev_results.json').read_text())
    report = {'version': version, 'mode': 'isolated ASGI replay; no listening server',
              'test_accessed': False, 'models': {}}
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        denied = await client.get(f'/api/neural-beacon/{version}/pairs', params={'split': 'test'})
        assert denied.status_code == 403
        denied_unknown = await client.post(f'/api/neural-beacon/{version}/score', json={'model': 'unknown', 'features': [[0.]]})
        assert denied_unknown.status_code == 404
        for kind, config in frozen['models'].items():
            offset, values, labels, features, decisions = 0, [], [], [], {k: [] for k in config['thresholds']}
            while True:
                response = await client.get(f'/api/neural-beacon/{version}/pairs', params={'offset': offset, 'limit': 128})
                response.raise_for_status()
                page = response.json()
                if not page['x']:
                    break
                response = await client.post(f'/api/neural-beacon/{version}/score', json={'model': kind, 'features': page['x']})
                response.raise_for_status()
                scored = response.json()
                assert scored['thresholds'] == config['thresholds']
                values.extend(scored['scores']); labels.extend(page['y']); features.extend(page['x'])
                for point in decisions:
                    decisions[point].extend(scored['accepted'][point])
                offset += len(page['x'])
                if offset >= page['total']:
                    break
            x, y, scores = np.asarray(features), np.asarray(labels), np.asarray(values)
            offline = -joblib.load(base / f'{kind}.joblib').decision_function(x[:, config['columns']])
            assert np.allclose(scores, offline, atol=1e-12, rtol=0)
            for point, threshold in config['thresholds'].items():
                assert np.array_equal(decisions[point], offline <= threshold)
            metrics = measure(y, scores, config['thresholds'])
            for point, rates in metrics['operating_points'].items():
                assert rates == stored['results'][kind]['operating_points'][point]
            assert abs(metrics['eer_diagnostic'] - stored['results'][kind]['eer_diagnostic']) < 1e-12
            report['models'][kind] = {'pairs_replayed': len(y), 'all_frozen_decisions_match': True,
                                     'max_absolute_score_error': float(np.max(abs(scores - offline))), 'metrics': metrics}
    external = httpx.ASGITransport(app=app, client=('10.0.0.2', 12345))
    async with httpx.AsyncClient(transport=external, base_url='http://127.0.0.1') as client:
        assert (await client.get(f'/api/neural-beacon/{version}/pairs')).status_code == 403
    report['test_guard_status'] = 403
    report['nonloopback_guard_status'] = 403
    with (base / 'api_replay.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v1', 'v2'], default='v1')
    args = parser.parse_args()
    if (ROOT / f'research/benchmarks/beacon-neural-{args.version}/api_replay.json').exists():
        raise SystemExit('Replay already recorded; preserve completed verification')
    asyncio.run(run(args.version))
