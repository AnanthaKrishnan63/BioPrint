import asyncio
import sys

from fastapi import FastAPI
import httpx
import numpy as np
import pytest

import research_type2branch_api as api


@pytest.fixture(autouse=True)
def isolate_frozen_failure(monkeypatch):
    monkeypatch.setattr(api,'FAILURE_REPORT_SHA256',None)


def test_failed_cohort_exposes_status_without_predictions(monkeypatch):
    failure={'status':'infeasible_prescribed_cohort','metrics':{},
             'failures':[{'subject':'generated','error':'missing window'}]}
    monkeypatch.setattr(api,'frozen_failure',lambda:failure)
    monkeypatch.setattr(api,'artifacts',forbidden)
    monkeypatch.setattr(api,'get_worker',forbidden)
    result=call('GET','/results').json()
    assert result['validation']['metrics']=={} and not result['scoring_available']
    assert call('GET','/dev/samples').status_code==409
    assert call('POST','/score',json={}).status_code==409
    assert call('GET','/test/samples').status_code==403


def call(method, path, **kwargs):
    app = FastAPI()
    app.include_router(api.router)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://localhost') as client:
            return await asyncio.wait_for(
                client.request(method, '/api/type2branch' + path, **kwargs), timeout=10)
    return asyncio.run(run())


def forbidden():
    raise AssertionError('Unexpected artifact or worker access')


def features():
    value = np.zeros((1, 100, 5), dtype=np.float32)
    value[:, :, 0] = np.float32(65 / 255)
    return value


@pytest.mark.parametrize('split', ['test', 'train', 'unknown'])
def test_nondev_rejected_before_artifacts(monkeypatch, split):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert call('GET', f'/{split}/samples').status_code == 403


def test_unreleased_is_unavailable(monkeypatch):
    monkeypatch.setattr(api, 'MANIFEST_SHA256', None)
    api._artifacts.cache_clear()
    assert call('GET', '/results').status_code == 503
    assert call('GET', '/dev/samples').status_code == 503


@pytest.mark.parametrize('bad', [True, '0', None, -1.6])
def test_invalid_features_before_artifacts_or_worker(monkeypatch, bad):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    monkeypatch.setattr(api, 'get_worker', forbidden)
    value = features().tolist()
    value[0][0][3] = bad
    assert call('POST', '/score', json={'features': value}).status_code == 422


@pytest.mark.parametrize('body', [{}, {'features': []}, {'features': [], 'path': '/x'}, []])
def test_bad_schema(monkeypatch, body):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert call('POST', '/score', json=body).status_code == 422


def test_malformed_json(monkeypatch):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert call('POST', '/score', content='{').status_code == 422


def test_valid_stub_routes(monkeypatch):
    accounts = [f'p{i:03d}' for i in range(79)]
    data = {'features': features(), 'subjects': np.array([accounts[0]]),
            'windows': np.array([0]), 'accounts': accounts, 'threshold': -.5,
            'report': {'status': 'generated_stub'}}
    monkeypatch.setattr(api, 'artifacts', lambda: data)
    class Worker:
        def score(self, value):
            assert value.dtype == np.float32 and value.shape == (1, 100, 5)
            return {'scores': [[-.25] * 79], 'accepted': [[True] * 79]}
    monkeypatch.setattr(api, 'get_worker', lambda: Worker())
    response = call('POST', '/score', json={'features': features().tolist()})
    assert response.status_code == 200
    assert response.json()['accounts'] == accounts
    assert response.json()['threshold'] == -.5
    assert response.json()['accepted'] == [[True] * 79]
    assert call('GET', '/dev/samples?limit=1').json()['windows'] == [0]
    assert call('GET', '/dev/samples?offset=10').json()['features'] == []
    assert call('GET', '/results').json() == data['report']


@pytest.mark.parametrize('query', ['limit=0', 'limit=33', 'offset=-1'])
def test_page_bounds(monkeypatch, query):
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert call('GET', '/dev/samples?' + query).status_code == 422


def test_worker_singleton_and_close(monkeypatch):
    import types
    created = []
    class Worker:
        def __init__(self, pin):
            self.pin = pin
            self.closed = False
            created.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setitem(sys.modules, 'type2branch_worker_client', types.SimpleNamespace(WorkerClient=Worker))
    monkeypatch.setattr(api, '_worker', None)
    monkeypatch.setattr(api, '_worker_pin', None)
    monkeypatch.setattr(api, 'MANIFEST_SHA256', 'a' * 64)
    first = api.get_worker()
    assert api.get_worker() is first
    monkeypatch.setattr(api, 'MANIFEST_SHA256', 'b' * 64)
    second = api.get_worker()
    assert first.closed and second is not first and len(created) == 2
    api.close_worker()
    api.close_worker()
    assert second.closed and api._worker is None
