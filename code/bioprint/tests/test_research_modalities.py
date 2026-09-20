"""Boundary tests use generated features, never sealed recordings."""
import asyncio

import httpx
import numpy as np
from research_api import app
import research_modalities


def request(method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


def test_sealed_split_and_invalid_shapes():
    assert request('GET', '/api/paired/beacon/test/samples').status_code == 403
    for features in [[[0.0]*32], [[0.0]*33, [0.0]*32], []]:
        response = request('POST', '/api/paired/beacon/models/equal_behavior/score',
                           json={'features': features})
        assert response.status_code == 422


def test_paired_route_inherits_loopback_guard():
    response = request('GET', '/api/paired/beacon/dev/samples',
                       headers={'Origin': 'http://192.168.1.2'})
    assert response.status_code == 403


def test_tabular_threshold_ties_and_shape(monkeypatch):
    class Model:
        def predict_proba(self, x):
            return np.column_stack([1-x[:, 0], x[:, 0]])

    monkeypatch.setattr(research_modalities, 'tabular_schema', lambda dataset: {
        'feature_count': 2, 'score_direction': 'higher_is_genuine',
        'models': {'generated': {'operating_points': {'0.01': {
            'threshold_selected_on_training_calibration': .5}}}}})
    monkeypatch.setattr(research_modalities, 'tabular_model', lambda dataset, name: Model())
    response = request('POST', '/api/features/device/models/generated/score',
                       json={'features': [[.49, 0], [.5, 0], [.51, 0]]})
    assert response.status_code == 200
    assert response.json()['accepted'] == [False, True, True]
    assert response.json()['scores'] == [.49, .5, .51]
    for features in [[[0]], [[0, 0], [0]]]:
        assert request('POST', '/api/features/device/models/generated/score',
                       json={'features': features}).status_code == 422
    assert request('GET', '/api/features/device/test/samples').status_code == 403
