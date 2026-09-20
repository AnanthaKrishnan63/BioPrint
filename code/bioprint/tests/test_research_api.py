"""Research API boundaries: no external access or sealed-test reads."""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/bioprint'))
from research_api import app


def request(path, address='127.0.0.1', host='http://127.0.0.1', **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app, client=(address, 12345))
        async with httpx.AsyncClient(transport=transport, base_url=host) as client:
            return await client.get(path, **kwargs)
    return asyncio.run(run())


@pytest.mark.parametrize('address', ['10.128.1.1', '192.168.1.10', '172.17.0.1'])
def test_rejects_nonloopback(address):
    assert request('/api/datasets', address=address).status_code == 403


def test_rejects_rebinding_and_foreign_origins():
    assert request('/api/datasets', host='http://attacker.example').status_code == 403
    assert request('/api/datasets', headers={'Origin': 'https://attacker.example'}).status_code == 403


@pytest.mark.parametrize('kind', ['raw', 'samples'])
def test_sealed_test_refused_before_dataset_lookup(kind):
    assert request(f'/api/datasets/unknown/test/{kind}').status_code == 403


def test_local_listing_and_noncacheable_dashboard():
    result = request('/api/datasets')
    assert result.status_code == 200
    assert result.json()['test'] == 'sealed'
    assert result.headers['Cache-Control'] == 'no-store'
    assert request('/').status_code == 200


def test_invalid_page_bounds_rejected():
    assert request('/api/datasets/keyrecs-fixed/train/samples?limit=99999').status_code == 422
    assert request('/api/datasets/keyrecs-fixed/train/samples?offset=-1').status_code == 422


def test_typenet_samples_keep_sequence_contract(monkeypatch):
    import numpy as np
    import research_api
    requested = []

    def fake_samples(dataset, split):
        requested.append((dataset, split))
        return np.zeros((1, 250)), np.array([0])

    monkeypatch.setattr(research_api, 'samples', fake_samples)
    monkeypatch.setattr(research_api, 'frozen', lambda _: {'subjects': ['synthetic'], 'feature_names': list(range(250))})
    response = request('/api/datasets/keyrecs-typenet/train/samples')
    assert response.status_code == 200
    assert len(response.json()['x'][0]) == 250
    assert requested == [('keyrecs-typenet', 'train')]


def test_neural_beacon_rejects_extreme_features_before_model_load(monkeypatch):
    import research_beacon_neural_api as neural
    monkeypatch.setattr(neural, 'configuration', lambda _: {'models': {'synthetic': {'columns': [0]}}})
    def forbidden(*args):
        raise AssertionError('Invalid input must not reach model loading')
    monkeypatch.setattr(neural, 'fitted', forbidden)
    async def run():
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
            return await client.post('/api/neural-beacon/v1/score', json={'model': 'synthetic', 'features': [[1e308]]})
    assert asyncio.run(run()).status_code == 422
