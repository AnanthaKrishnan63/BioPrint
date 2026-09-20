"""Replay public pointer dev measurements through ASGI, without any listener."""
import argparse
import asyncio
import json
import pathlib
import sys
import httpx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
from research_api import app
from research_pointer_api import router, spec

# Parent may already include the router. Include locally only when absent.
if not any(getattr(route, 'path', '') == '/api/pointer/datasets' for route in app.routes):
    app.include_router(router)


async def replay(dataset):
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        sealed = await client.get(f'/api/pointer/{dataset}/test/samples')
        assert sealed.status_code == 403, sealed.text
        x, sequence, labels, groups = [], [], [], []
        offset, total = 0, 1
        while offset < total:
            response = await client.get(f'/api/pointer/{dataset}/dev/samples', params={'offset': offset, 'limit': 128})
            response.raise_for_status()
            page = response.json()
            x.extend(page['features'])
            if page['sequences'] is not None:
                sequence.extend(page['sequences'])
            labels.extend(page['labels'])
            groups.extend(page['session_ids'])
            total = page['total']
            offset += len(page['features'])
        config = spec(dataset)
        checks = {}
        for model, method in config['methods'].items():
            body = {'model': model, 'features': x, 'sequences': sequence or None, 'session_ids': groups}
            response = await client.post(f'/api/pointer/{dataset}/score', json=body)
            response.raise_for_status()
            got = response.json()
            expected_path = config['source'] / f'{method.replace(":", "_")}_dev_scores.npz'
            with np.load(expected_path, allow_pickle=False) as archive:
                expected = archive['scores'].T
                expected_users = archive['users'].tolist()
                expected_labels = archive['true_user'].tolist()
            scores = np.asarray(got['scores'])
            assert got['subjects'] == expected_users
            assert [labels[i] for i in got['input_start_indices']] == expected_labels
            np.testing.assert_allclose(scores, expected, rtol=2e-5, atol=2e-6)
            threshold = np.asarray(got['thresholds'])
            np.testing.assert_array_equal(np.asarray(got['accepted']), expected >= threshold)
            actual = np.asarray(expected_labels)
            users = np.asarray(expected_users)
            truth = actual[:, None] == users[None, :]
            accepted = np.asarray(got['accepted'])
            far = float(accepted[~truth].mean())
            frr = float((~accepted[truth]).mean())
            stored = json.loads((config['source'] / 'results.json').read_text())['dev'][method]['0.01']
            np.testing.assert_allclose([far, frr], [stored['far'], stored['frr']], atol=1e-12)
            checks[model] = {'method': method, 'score_shape': list(scores.shape), 'maximum_absolute_score_difference': float(np.abs(scores - expected).max()), 'far': far, 'frr': frr, 'threshold_decisions_match': True}
        malformed = await client.post(f'/api/pointer/{dataset}/score', json={'model': 'baseline', 'features': [[1], [1, 2]], 'session_ids': ['x', 'x']})
        assert malformed.status_code == 422, malformed.text
        oversized = await client.post(f'/api/pointer/{dataset}/score', json={'model': 'baseline', 'features': [[1e100] * 29], 'session_ids': ['synthetic']})
        assert oversized.status_code == 422, oversized.text
        cross_origin = await client.get('/api/pointer/datasets', headers={'origin': 'https://untrusted.example'})
        assert cross_origin.status_code == 403
    remote = httpx.ASGITransport(app=app, client=('192.0.2.1', 12345))
    async with httpx.AsyncClient(transport=remote, base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/pointer/datasets')).status_code == 403
    result = {'dataset': dataset, 'input_dev_windows': len(labels), 'models': checks, 'sealed_test_http_status': 403, 'remote_access_http_status': 403, 'cross_origin_http_status': 403, 'malformed_input_http_status': 422, 'oversized_numeric_input_http_status': 422, 'network_listener_opened': False, 'input_source': 'dev measurements requested from API; actual models infer from submitted features/sequences'}
    target = ROOT / 'research/benchmarks' / f'pointer_{dataset}_api_replay.json'
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['balabit', 'sapimouse'])
    asyncio.run(replay(parser.parse_args().dataset))
