import asyncio
import json
import sys
import types

from fastapi import FastAPI, HTTPException
import httpx
import numpy as np
import pytest

import research_type2branch_capture_api as api


def call(method, path, **kwargs):
    app = FastAPI(); app.include_router(api.router)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            return await asyncio.wait_for(client.request(method, '/api/type2branch-capture' + path, **kwargs), 10)
    return asyncio.run(run())


def forbidden():
    raise AssertionError('Unexpected artifact/worker access')


def generated_release(tmp_path, monkeypatch, count=2):
    accounts = [f'p{i:03d}' for i in range(79)]
    x = np.zeros((count, 100, 5), dtype=np.float32)
    lengths = np.full(count, 25, dtype=np.int64)
    np.savez_compressed(tmp_path/'dev_features.npz', features=x, subject=np.array(accounts[:count], dtype=str),
                        window=np.zeros(count, dtype=np.int64), true_length=lengths)
    coverage = {'eligible': count, 'total': 79, 'ineligible_count': 79-count,
                'ineligible': [{'identity': s, 'reason': 'short'} for s in accounts[count:]],
                'by_length': {str(n): {'eligible': count if n == 25 else 0} for n in (25, 50, 75, 100)}}
    report = {'status': 'frozen_capture_dev_evaluation_complete',
              'metrics': {'coverage': coverage, 'conditional_metrics': {}}}
    (tmp_path/'dev_report.json').write_text(json.dumps(report))
    manifest = {'subjects': accounts, 'thresholds': {str(n): -.5 for n in (25, 50, 75, 100)},
                'eligible_probes': count,
                'files': {n: api.digest((tmp_path/n).read_bytes()) for n in ('dev_features.npz', 'dev_report.json')}}
    monkeypatch.setattr(api, 'RELEASE', tmp_path)
    monkeypatch.setattr(api, 'verify_release', lambda pin: manifest)
    monkeypatch.setattr(api, 'MANIFEST_SHA256', 'a' * 64)
    api._artifacts.cache_clear()
    return manifest, report


@pytest.mark.parametrize('split', ['test', 'train', 'unknown'])
def test_nondev_before_loading(monkeypatch, split):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert call('GET', f'/{split}/samples').status_code == 403


def test_pending_release(monkeypatch):
    monkeypatch.setattr(api, 'MANIFEST_SHA256', None)
    api._artifacts.cache_clear()
    assert call('GET', '/results').status_code == 503


@pytest.mark.parametrize('fault', ['bool', 'string', 'tail', 'tiny_tail', 'length_bool', 'length_missing', 'extra'])
def test_invalid_score_before_loading(monkeypatch, fault):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    monkeypatch.setattr(api, 'get_worker', forbidden)
    body = {'features': np.zeros((1, 100, 5)).tolist(), 'true_lengths': [25]}
    if fault == 'bool': body['features'][0][0][3] = True
    if fault == 'string': body['features'][0][0][3] = '0'
    if fault == 'tail': body['features'][0][25][3] = .1
    if fault == 'tiny_tail': body['features'][0][25][3] = 1e-100
    if fault == 'length_bool': body['true_lengths'] = [True]
    if fault == 'length_missing': del body['true_lengths']
    if fault == 'extra': body['path'] = '/x'
    assert call('POST', '/score', json=body).status_code == 422


def test_generated_release_and_stub_inference(tmp_path, monkeypatch):
    manifest, report = generated_release(tmp_path, monkeypatch)
    class Worker:
        def score(self, features, lengths):
            assert features.dtype == np.float32 and lengths == [25, 25]
            return {'scores': [[0.] * 79] * 2, 'accepted': [[True] * 79] * 2}
    monkeypatch.setattr(api, 'get_worker', lambda: Worker())
    page = call('GET', '/dev/samples').json()
    assert page['total'] == 2 and page['true_lengths'] == [25, 25]
    response = call('POST', '/score', json={name: page[name] for name in ('features', 'true_lengths')})
    assert response.status_code == 200
    assert response.json()['accounts'] == manifest['subjects']
    assert response.json()['thresholds'] == manifest['thresholds']
    assert call('GET', '/results').json() == report


def test_all_ineligible_release(tmp_path, monkeypatch):
    generated_release(tmp_path, monkeypatch, count=0)
    monkeypatch.setattr(api, 'get_worker', forbidden)
    page = call('GET', '/dev/samples').json()
    assert page['total'] == 0 and page['features'] == page['true_lengths'] == []
    assert call('GET', '/results').json()['metrics']['conditional_metrics'] == {}


def test_tamper_before_archive_decode(tmp_path, monkeypatch):
    generated_release(tmp_path, monkeypatch)
    (tmp_path/'dev_features.npz').write_bytes(b'broken')
    with pytest.raises(HTTPException) as error: api.artifacts()
    assert error.value.status_code == 503


def test_coverage_count_mismatch(tmp_path, monkeypatch):
    manifest, _ = generated_release(tmp_path, monkeypatch)
    manifest['eligible_probes'] = 3
    with pytest.raises(HTTPException) as error: api.artifacts()
    assert error.value.status_code == 503


def test_worker_singleton_and_shutdown(monkeypatch):
    created = []
    class Worker:
        def __init__(self, pin): self.closed = False; created.append(self)
        def close(self): self.closed = True
    monkeypatch.setitem(sys.modules, 'type2branch_capture_worker_client', types.SimpleNamespace(WorkerClient=Worker))
    monkeypatch.setattr(api, '_worker', None); monkeypatch.setattr(api, '_worker_pin', None)
    monkeypatch.setattr(api, 'MANIFEST_SHA256', 'a' * 64)
    first = api.get_worker(); assert api.get_worker() is first
    monkeypatch.setattr(api, 'MANIFEST_SHA256', 'b' * 64)
    second = api.get_worker(); assert first.closed and second is not first
    api.close_worker(); api.close_worker()
    assert second.closed and len(created) == 2
