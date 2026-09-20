"""Generated-only HMOG endpoint checks; no frozen research observations."""
import asyncio
import hashlib
import json

from fastapi import FastAPI, HTTPException
import httpx
import numpy as np
import pytest

import research_hmog_api as api


def request(method, url, **kwargs):
    async def run():
        app = FastAPI()
        app.include_router(api.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://127.0.0.1') as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(run())


def test_test_route_denied_before_artifacts(monkeypatch):
    def forbidden():
        raise AssertionError('Sealed route must not touch artifacts')
    monkeypatch.setattr(api, 'artifacts', forbidden)
    assert request('GET', '/api/hmog/test/samples').status_code == 403


def test_unfrozen_release_unavailable(monkeypatch):
    api.artifacts.cache_clear()
    monkeypatch.setattr(api, 'MANIFEST_SHA256', None)
    assert request('GET', '/api/hmog/dev/samples').status_code == 503


@pytest.mark.parametrize('key,imu', [
    ([], []), (np.zeros((1, 49, 10)), np.zeros((1, 100, 24))),
    (np.zeros((9, 50, 10)), np.zeros((9, 100, 24))),
    (np.zeros((1, 50, 10)), np.zeros((2, 100, 24))),
    (np.full((1, 50, 10), np.nan), np.zeros((1, 100, 24))),
    (np.zeros((1, 50, 10)), np.full((1, 100, 24), np.inf)),
    (np.full((1, 50, 10), 1e100), np.zeros((1, 100, 24))),
])
def test_invalid_input_rejected(key, imu):
    with pytest.raises(HTTPException) as error:
        api.input_arrays(key, imu)
    assert error.value.status_code == 422


def test_shape_rejection_precedes_artifacts(monkeypatch):
    def forbidden():
        raise AssertionError('Malformed input must not load model')
    monkeypatch.setattr(api, 'artifacts', forbidden)
    response = request('POST', '/api/hmog/score', json={'key': [[[1]]], 'imu': [[[1]]]})
    assert response.status_code == 422


def test_corrupt_release_denied_before_decoding(tmp_path, monkeypatch):
    manifest = {'version': 1, 'split': 'dev',
                'files': {name: hashlib.sha256(b'expected').hexdigest() for name in api.FILES},
                'sources': {name: '0' * 64 for name in api.SOURCES}}
    raw = json.dumps(manifest).encode()
    (tmp_path / 'manifest.json').write_bytes(raw)
    for name in api.FILES:
        (tmp_path / name).write_bytes(b'corrupt')
    monkeypatch.setattr(api, 'RELEASE', tmp_path)
    monkeypatch.setattr(api, 'MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    api.artifacts.cache_clear()
    with pytest.raises(HTTPException) as error:
        api.artifacts()
    assert error.value.status_code == 503
    assert 'checksum' in str(error.value.__cause__).lower()


def test_score_endpoint_all_branches_and_inclusive_thresholds(monkeypatch):
    import torch
    import hmog_verification
    data = {'accounts': ['a', 'b'], 'model': object(),
            'profiles': {b: np.array([np.zeros(64), np.ones(64)]) for b in api.BRANCHES},
            'thresholds': {b: {'0.001': 0., '0.01': 7.9, '0.05': 8.} for b in api.BRANCHES}}
    monkeypatch.setattr(api, 'artifacts', lambda: data)
    monkeypatch.setattr(hmog_verification, 'branch_embeddings',
                        lambda model, key, imu: {b: torch.zeros((len(key), 64)) for b in api.BRANCHES})
    response = request('POST', '/api/hmog/score', json={
        'key': np.zeros((1, 50, 10)).tolist(), 'imu': np.zeros((1, 100, 24)).tolist()})
    assert response.status_code == 200
    result = response.json()
    for branch in api.BRANCHES:
        assert result['branches'][branch]['distances'] == [[0., 8.]]
        assert result['branches'][branch]['accepted']['0.001'] == [[True, False]]
        assert result['branches'][branch]['accepted']['0.05'] == [[True, True]]


def make_generated_release(tmp_path, monkeypatch, *, subject='556357', session=9,
                           role='dev_probe', poison_features=False):
    """Build a fully hash-bound release using generated arrays, never observations."""
    accounts = np.array(sorted(('556357', '219303', '777078', '737973')))
    np.savez(tmp_path / 'profiles.npz', accounts=accounts,
             **{branch: np.zeros((4, 64)) for branch in api.BRANCHES})
    key = (np.array([object()], dtype=object) if poison_features
           else np.zeros((1, 50, 10), np.float32))
    np.savez(tmp_path / 'dev_features.npz', key=key,
             imu=np.zeros((1, 100, 24), np.float32),
             subject=np.array([subject]), session=np.array([session]), role=np.array([role]))
    (tmp_path / 'encoder.pt').write_bytes(b'generated checkpoint placeholder')
    (tmp_path / 'thresholds.json').write_text(json.dumps({
        branch: {target: 1. for target in api.TARGETS} for branch in api.BRANCHES}))
    (tmp_path / 'dev_report.json').write_text('{"generated_only": true}')
    manifest = {'version': 1, 'split': 'dev',
                'files': {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                          for name in api.FILES},
                'sources': {name: hashlib.sha256((api.ROOT / 'scripts' / name).read_bytes()).hexdigest()
                            for name in api.SOURCES}}
    raw = json.dumps(manifest).encode()
    (tmp_path / 'manifest.json').write_bytes(raw)
    monkeypatch.setattr(api, 'RELEASE', tmp_path)
    monkeypatch.setattr(api, 'MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    api.artifacts.cache_clear()


@pytest.mark.parametrize('subject,session,role', [
    ('556357', 1, 'train_support'), ('556357', 8, 'train_enrollment'),
    ('556357', 17, 'dev_probe'), ('556357', 9, 'sealed_test'),
    ('717868', 9, 'dev_probe'), ('180679', 9, 'dev_probe'),
    ('556357', 9.0, 'dev_probe'),
])
def test_hash_bound_release_rejects_forbidden_metadata_before_values_or_checkpoint(
        tmp_path, monkeypatch, subject, session, role):
    import torch
    import hmog_training_core
    make_generated_release(tmp_path, monkeypatch, subject=subject, session=session,
                           role=role, poison_features=True)
    def forbidden(*args, **kwargs):
        pytest.fail('Forbidden metadata reached model loading/deserialization')
    monkeypatch.setattr(hmog_training_core, 'load_author_classes', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(HTTPException) as error:
        api.artifacts()
    assert error.value.status_code == 503
    # Pickle-backed key array would fail if copied before the metadata guard.
    assert str(error.value.__cause__) == 'Only designated DEV-probe rows may be exposed'


def test_valid_hash_bound_dev_metadata_reaches_model_loader(tmp_path, monkeypatch):
    import torch
    import hmog_training_core
    make_generated_release(tmp_path, monkeypatch)
    called = []
    def unavailable_author_model():
        called.append(True)
        raise RuntimeError('generated unavailable-model sentinel')
    monkeypatch.setattr(hmog_training_core, 'load_author_classes', unavailable_author_model)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Model unavailable before deserialization'))
    with pytest.raises(HTTPException) as error:
        api.artifacts()
    assert error.value.status_code == 503
    assert called == [True]
    assert str(error.value.__cause__) == 'generated unavailable-model sentinel'


def test_hash_bound_non_mapping_checkpoint_fails_integrity_cleanly(tmp_path, monkeypatch):
    import torch
    import hmog_training_core
    make_generated_release(tmp_path, monkeypatch)
    monkeypatch.setattr(hmog_training_core, 'load_author_classes',
                        lambda: (lambda *args: torch.nn.Linear(1, 1), None))
    def generated_non_mapping_checkpoint(stream, *, weights_only, map_location):
        assert weights_only is True and map_location == 'cpu'
        return []
    monkeypatch.setattr(torch, 'load', generated_non_mapping_checkpoint)
    with pytest.raises(HTTPException) as error:
        api.artifacts()
    assert error.value.status_code == 503
    assert str(error.value.__cause__) == 'Nonfinite checkpoint'
