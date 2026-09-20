"""Export the frozen comparison and replay real DEV inputs through ASGI."""
import argparse
import asyncio
import json
import numpy as np
import httpx
from pointer_sapimouse_znorm_dev import ROOT, OUT, digest


def export():
    report_path = OUT / 'dev_complete.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete_evaluation_only' or report['candidate_promoted'] is not False:
        raise ValueError('Frozen comparison only; no candidate promotion')
    files = dict(report['plan']['input_sha256'])
    for relative, expected in files.items():
        if digest(ROOT / relative) != expected: raise ValueError('Frozen comparison input changed')
    for name, expected in report['plan']['source_sha256'].items():
        p = ROOT / 'scripts' / name
        if digest(p) != expected: raise ValueError('Frozen comparison source changed')
        files[str(p.relative_to(ROOT))] = expected
    if digest(OUT / 'normalization.npz') != report['normalization_sha256']:
        raise ValueError('Normalization changed')
    if digest(OUT / 'scores.npz') != report['scores_sha256']: raise ValueError('Expected scores changed')
    paths = [report_path, OUT / 'normalization.npz', OUT / 'scores.npz',
             ROOT / 'research/benchmarks/pointer_sapimouse/encoder.torchscript.pt',
             ROOT / 'research/benchmarks/pointer_sapimouse/encoder_manifest.json',
             ROOT / 'code/bioprint/research_pointer_api.py', ROOT / 'code/bioprint/engine/pointer_sequence.py']
    files.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    release = {'files': files, 'thresholds': report['plan']['thresholds'], 'comparison': report,
               'training_selected_method': 'cosine@5', 'candidate_promoted': False}
    with (OUT / 'release.json').open('x') as f: json.dump(release, f, indent=2, allow_nan=False)
    print(digest(OUT / 'release.json'))


async def replay():
    from research_api import app
    from research_pointer_znorm_api import RELEASE_SHA256
    if digest(OUT / 'release.json') != RELEASE_SHA256: raise ValueError('API release pin mismatch')
    report = json.loads((OUT / 'dev_complete.json').read_text())
    if digest(OUT / 'scores.npz') != report['scores_sha256']: raise ValueError('Expected scores changed')
    with np.load(OUT / 'scores.npz', allow_pickle=False) as a:
        expected = {k: a[k].copy() for k in a.files}
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    checks = {}
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/pointer-znorm/test/samples')).status_code == 403
        sequence, sessions, labels = [], [], []
        offset, total = 0, 1
        while offset < total:
            response = await client.get('/api/pointer-znorm/dev/samples', params={'offset': offset, 'limit': 128})
            response.raise_for_status(); page = response.json()
            if not page['sequences']: raise ValueError('Unexpected empty DEV page')
            sequence.extend(page['sequences']); sessions.extend(page['session_ids']); labels.extend(page['labels'])
            total = page['total']; offset = len(sequence)
        assert len(sequence) == report['dev_blocks']
        for method, key in [('cosine', 'cosine@5'), ('znorm', 'znorm_cosine@5')]:
            response = await client.post('/api/pointer-znorm/score', json={
                'method': method, 'sequences': sequence, 'session_ids': sessions})
            response.raise_for_status(); got = response.json()
            values = np.asarray(got['scores'])
            wanted = expected[method].T
            tol = report['plan']['api_tolerance']
            np.testing.assert_allclose(values, wanted, rtol=tol['rtol'],
                                       atol=tol['raw_atol' if method == 'cosine' else 'znorm_atol'])
            assert got['subjects'] == expected['accounts'].tolist()
            assert got['session_ids'] == expected['sessions'].tolist()
            actual = np.asarray(labels)[got['input_start_indices']]
            np.testing.assert_array_equal(actual, expected['actual'])
            truth = actual[:, None] == expected['accounts'][None, :]
            for target, threshold in report['plan']['thresholds'][key].items():
                accepted = np.asarray(got['accepted_at_targets'][target])
                np.testing.assert_array_equal(accepted, wanted >= threshold)
                rates = report['dev'][key][target]
                np.testing.assert_allclose([accepted[~truth].mean(), (~accepted[truth]).mean()],
                                           [rates['far'], rates['frr']], atol=0, rtol=0)
            assert not got['candidate_promoted'] and got['training_selected_method'] == 'cosine@5'
            checks[method] = {'scores': values.size, 'maximum_absolute_error': float(np.abs(values - wanted).max()),
                              'all_three_threshold_decisions_match': True}
        response = await client.get('/api/pointer-znorm/results'); response.raise_for_status()
        assert response.json() == report
        assert (await client.get('/api/pointer-znorm/results', headers={'origin': 'https://untrusted.example'})).status_code == 403
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=('192.0.2.1', 1)), base_url='http://local') as client:
        assert (await client.get('/api/pointer-znorm/results')).status_code == 403
    result = {'status': 'complete', 'release_sha256': RELEASE_SHA256, 'models': checks,
              'source_sha256': digest(ROOT / 'code/bioprint/research_pointer_znorm_api.py'),
              'network_listener_opened': False, 'test_route_status': 403, 'candidate_promoted': False}
    with (OUT / 'api_replay.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['export', 'replay'])
    a = p.parse_args(); export() if a.action == 'export' else asyncio.run(replay())
