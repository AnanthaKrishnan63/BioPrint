import asyncio
import json
import joblib
import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import research_beacon_type2branch_api as api


class GeneratedModel:
    def decision_function(self, x):
        return x[:, 0]


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    train = tmp_path/'train'; dev = tmp_path/'dev'
    train.mkdir(); dev.mkdir()
    models = {}
    for i in range(8):
        name = f'model{i}'
        joblib.dump(GeneratedModel(), train/f'{name}.joblib')
        models[name] = {'columns': [i], 'thresholds': {'far_1pct': 0., 'far_5pct': 1., 'eer': .5},
                        'model_sha256': api.digest(train/f'{name}.joblib')}
    config = {'status': 'paired_train_fusion_complete', 'models': models,
              'protocol': {'models': {n: v['columns'] for n, v in models.items()}}}
    (train/'frozen.json').write_text(json.dumps(config))
    frozen_sha = api.digest(train/'frozen.json')
    np.savez_compressed(dev/'dev_pairs.npz', X=np.zeros((4, 35)), y=np.array([1, 0, 0, 1]),
                        groups=np.array([['a','a','0'],['a','b','0'],['b','a','0'],['b','b','0']]))
    report = {'status': 'paired_dev_evaluation_complete', 'evaluation_split': 'dev',
              'frozen_sha256': frozen_sha, 'results': {n: {} for n in models},
              'pairs_sha256': api.digest(dev/'dev_pairs.npz'), 'claims': 4, 'subjects': ['a','b']}
    (dev/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(api, 'TRAIN', train); monkeypatch.setattr(api, 'DEV', dev)
    monkeypatch.setattr(api, 'FROZEN_SHA256', frozen_sha)
    monkeypatch.setattr(api, 'REPORT_SHA256', api.digest(dev/'report.json'))
    return train, dev


def request(method, path, **kwargs):
    async def run():
        app = FastAPI(); app.include_router(api.router)
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            return await client.request(method, '/api/type2branch-beacon'+path, **kwargs)
    return asyncio.run(run())


def test_actual_generated_model_inclusive_score_direction(artifacts):
    rows = np.zeros((3, 35)); rows[:, 0] = [0., 1., -1.]
    response = request('POST', '/score', json={'model':'model0','features':rows.tolist()})
    assert response.status_code == 200
    result = response.json()
    assert result['scores'] == [0., -1., 1.]
    assert result['accepted']['far_1pct'] == [True, True, False]
    assert result['accepted']['far_5pct'] == [True, True, True]
    assert 'offline' in result['boundary']


def test_pagination_info_results(artifacts):
    response = request('GET','/pairs?offset=1&limit=2')
    assert response.status_code == 200
    assert response.json()['total'] == 4 and response.json()['y'] == [0, 0]
    assert request('GET','/info').json()['feature_width'] == 35
    assert request('GET','/results').json()['claims'] == 4


@pytest.mark.parametrize('split',['train','test'])
def test_split_denial_precedes_artifact_loading(monkeypatch, split):
    monkeypatch.setattr(api,'artifacts',lambda: pytest.fail('Forbidden split loaded artifacts'))
    assert request('GET',f'/pairs?split={split}').status_code == 403


@pytest.mark.parametrize('route',['/info','/results','/pairs'])
def test_missing_pin_fails_closed(monkeypatch,route):
    monkeypatch.setattr(api,'REPORT_SHA256',None)
    assert request('GET',route).status_code == 503


@pytest.mark.parametrize('filename',['frozen.json','model0.joblib','report.json','dev_pairs.npz'])
def test_integrity_failure(artifacts,filename):
    train, dev = artifacts
    path = (train if filename in ['frozen.json','model0.joblib'] else dev)/filename
    with path.open('ab') as stream:stream.write(b'changed')
    assert request('GET','/results').status_code == 503


@pytest.mark.parametrize('features',[[], [[0.]*34], [[1e100]*35], [[0.]*35]*257])
def test_bad_feature_inputs(artifacts, features):
    assert request('POST','/score',json={'model':'model0','features':features}).status_code == 422


def test_unknown_model_and_bad_pagination(artifacts):
    assert request('POST','/score',json={'model':'absent','features':[[0.]*35]}).status_code == 404
    assert request('GET','/pairs?limit=257').status_code == 422
    assert request('GET','/pairs?offset=-1').status_code == 422
