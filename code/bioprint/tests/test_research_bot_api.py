"""Synthetic guard checks; public-data parity is in bot_api_replay.py."""
import asyncio
import pytest
from fastapi import HTTPException
import research_bot_api as api


def test_test_split_rejected_before_measurements(monkeypatch):
    def forbidden():
        raise AssertionError('Sealed requests must not load measurements')
    monkeypatch.setattr(api, 'development', forbidden)
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.samples(split='test', offset=0, limit=1))
    assert error.value.status_code == 403


def test_unknown_owner_rejected(monkeypatch):
    monkeypatch.setattr(api, 'priors', lambda: {})
    body = api.CheckRequest(rows=[{'claimed_owner': 'unknown', 'sample': {'keystrokes': []}}])
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.infer(body))
    assert error.value.status_code == 404


@pytest.mark.parametrize('change', [{'env': {'webdriver': True}}, {'keystrokes': [
    {'code': 'KeyA', 'type': 'down', 't': 1e308, 'field': 'password'}]}])
def test_out_of_contract_input_rejected(monkeypatch, change):
    monkeypatch.setattr(api, 'priors', lambda: {'synthetic': []})
    body = api.CheckRequest(rows=[{'claimed_owner': 'synthetic', 'sample': {'keystrokes': [], **change}}])
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.infer(body))
    assert error.value.status_code == 422
