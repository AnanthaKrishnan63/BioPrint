"""Verify serialized device/bot/cognitive models through the isolated ASGI API."""
import argparse
import asyncio
import json

import httpx
import numpy as np
from research_api import app
from research_modalities import artifact_directory, tabular_schema, tabular_model


async def replay(dataset, output=None):
    directory = artifact_directory(dataset)
    destination = output or directory / 'api_replay.json'
    if destination.exists():
        raise SystemExit(f'{destination} exists; inspect logs before repeating')
    schema = tabular_schema(dataset)
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        assert (await client.get(f'/api/features/{dataset}/test/samples')).status_code == 403
        features, labels, offset = [], [], 0
        while True:
            response = await client.get(f'/api/features/{dataset}/dev/samples', params={'offset': offset, 'limit': 256})
            response.raise_for_status()
            page = response.json()
            features.extend(page['features']); labels.extend(page['genuine'])
            offset += len(page['features'])
            if offset >= page['total']:
                break
        x, y = np.asarray(features), np.asarray(labels)
        results = {}
        for name, spec in schema['models'].items():
            scores, accepted = [], []
            threshold = spec['operating_points']['0.01']['threshold_selected_on_training_calibration']
            for start in range(0, len(x), 256):
                response = await client.post(f'/api/features/{dataset}/models/{name}/score',
                                             json={'features': x[start:start+256].tolist()})
                response.raise_for_status()
                body = response.json()
                assert body['threshold'] == threshold
                assert body['score_direction'] == schema['score_direction']
                np.testing.assert_array_equal(body['accepted'], np.asarray(body['scores']) >= threshold)
                scores.extend(body['scores']); accepted.extend(body['accepted'])
            expected = tabular_model(dataset, name).predict_proba(x)[:, 1]
            np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
            accepted = np.asarray(accepted)
            far, frr = float(accepted[y == 0].mean()), float((~accepted[y == 1]).mean())
            reported = spec['operating_points']['0.01']['dev']
            np.testing.assert_allclose([far, frr], [reported['far'], reported['frr']], atol=1e-12)
            results[name] = {'max_score_error': float(np.max(abs(np.asarray(scores)-expected))),
                             'far': far, 'frr': frr}
    report = {'dataset': dataset, 'dev_samples': len(x), 'models': results,
              'test_access_denied': True, 'network_listener_started': False,
              'production_database_used': False}
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['device', 'delbot', 'cognitive', 'touch_tsi', 'beacon_nonlinear'])
    from pathlib import Path
    parser.add_argument('--output', type=Path, help='New report path; existing files are never overwritten')
    args = parser.parse_args()
    asyncio.run(replay(args.dataset, args.output))
