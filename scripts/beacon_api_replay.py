"""Replay every frozen BEACON dev pair through ASGI, without a network listener."""
import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
from research_api import app
from research_modalities import BEACON, configuration, model_for
from beacon_benchmark import measure


async def main(output=None):
    destination = output or BEACON / 'api_replay.json'
    if destination.exists():
        raise SystemExit('Replay exists; inspect logs before repeating')
    config = configuration()
    offline = json.loads((BEACON / 'dev_results.json').read_text())['results']
    reports = {}
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/paired/beacon/test/samples')).status_code == 403
        features, labels = [], []
        offset = 0
        while True:
            response = await client.get('/api/paired/beacon/dev/samples', params={'offset': offset})
            response.raise_for_status()
            page = response.json()
            features.extend(page['features']); labels.extend(page['genuine'])
            offset += len(page['features'])
            if offset >= page['total']:
                break
        x, y = np.asarray(features), np.asarray(labels)
        for name in [*config['models'], 'equal_behavior']:
            scores = []
            accepted = []
            thresholds = (config['baseline_thresholds'] if name == 'equal_behavior'
                          else config['models'][name]['thresholds'])
            for start in range(0, len(x), 128):
                response = await client.post(f'/api/paired/beacon/models/{name}/score',
                                             json={'features': x[start:start+128].tolist()})
                response.raise_for_status()
                body = response.json()
                assert body['threshold'] == -thresholds['far_1pct']
                assert body['score_direction'] == 'higher_is_more_genuine'
                np.testing.assert_array_equal(body['accepted'], np.asarray(body['scores']) >= body['threshold'])
                scores.extend(body['scores'])
                accepted.extend(body['accepted'])
            distances = -np.asarray(scores)
            if name == 'equal_behavior':
                expected = x[:, :32].mean(axis=1)
                thresholds = config['baseline_thresholds']
            else:
                spec = config['models'][name]
                expected = -model_for(name).decision_function(x[:, spec['columns']])
                thresholds = spec['thresholds']
            np.testing.assert_allclose(distances, expected, atol=1e-12, rtol=1e-12)
            measured = measure(y, distances, thresholds)
            reference = offline['untrained_equal_behavior_fusion' if name == 'equal_behavior' else name]
            assert measured['genuine_comparisons'] == reference['genuine_comparisons']
            assert measured['impostor_comparisons'] == reference['impostor_comparisons']
            np.testing.assert_allclose(measured['eer_diagnostic'], reference['eer_diagnostic'], atol=1e-12, rtol=0)
            for point, rates in measured['operating_points'].items():
                np.testing.assert_allclose([rates['far'], rates['frr']],
                    [reference['operating_points'][point]['far'], reference['operating_points'][point]['frr']],
                    atol=1e-12, rtol=0)
            decisions = np.asarray(accepted, dtype=bool)
            np.testing.assert_allclose([decisions[y == 0].mean(), (~decisions[y == 1]).mean()],
                [reference['operating_points']['far_1pct']['far'], reference['operating_points']['far_1pct']['frr']],
                atol=1e-12, rtol=0)
            reports[name] = {'max_score_error': float(np.max(abs(distances-expected))),
                             'metrics': measured, 'offline_report_and_acceptance_parity': True}
    report = {'pairs': len(x), 'models': reports, 'test_access_denied': True,
              'network_listener_started': False, 'production_database_used': False}
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='New report path; existing files are never overwritten')
    asyncio.run(main(parser.parse_args().output))
