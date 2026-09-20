import asyncio
import httpx
from fastapi import FastAPI, HTTPException
import pytest
import research_arithmetic_api as api


def test_test_route_rejects_before_artifact_access(monkeypatch):
    monkeypatch.setattr(api, 'artifacts', lambda: (_ for _ in ()).throw(AssertionError('Artifacts accessed')))
    app = FastAPI(); app.include_router(api.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            assert (await client.get('/api/arithmetic/test/samples')).status_code == 403
    asyncio.run(check())


def test_infeasible_data_and_scoring_never_report_zero_errors(monkeypatch):
    failure = {'status': 'infeasible', 'metrics': {}, 'invalid_blocks': [{'member': 'bad.mat'}]}
    train = {'selected': 'example', 'selection': {}, 'calibration': {}}
    monkeypatch.setattr(api, 'artifacts', lambda: (train, failure))
    app = FastAPI(); app.include_router(api.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            result = (await client.get('/api/arithmetic/results')).json()
            assert result['validation']['metrics'] == {}
            assert not result['validation_metrics_available'] and not result['scoring_available']
            assert (await client.get('/api/arithmetic/dev/samples')).status_code == 409
            assert (await client.post('/api/arithmetic/score', json={})).status_code == 409
    asyncio.run(check())


def test_tampered_artifact_rejected_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'BASE', tmp_path)
    monkeypatch.setattr(api, 'PINS', {'bad.json': '0' * 64})
    (tmp_path / 'bad.json').write_text('not json')
    with pytest.raises(HTTPException) as exc: api.artifacts()
    assert exc.value.status_code == 503
