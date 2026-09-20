import asyncio
import httpx
import numpy as np
from fastapi import FastAPI
import research_pointer_znorm_api as api


def test_sealed_samples_rejected_before_artifact_access(monkeypatch):
    monkeypatch.setattr(api, 'artifacts', lambda: (_ for _ in ()).throw(AssertionError('Artifact accessed')))
    app = FastAPI(); app.include_router(api.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            for split in ['test', 'train']:
                assert (await client.get(f'/api/pointer-znorm/{split}/samples')).status_code == 403
    asyncio.run(check())


def test_frozen_normalization_and_all_operating_decisions(monkeypatch):
    thresholds = {'0.001': 3., '0.01': 1., '0.05': .5}
    manifest = {'thresholds': {'znorm_cosine@5': thresholds, 'cosine@5': thresholds}}
    monkeypatch.setattr(api, 'artifacts', lambda: (manifest, ['a', 'b'], np.array([2., 10.]), np.array([2., 2.])))
    async def original(dataset, body):
        assert dataset == 'sapimouse' and body.model == 'selected'
        return {'subjects': ['a', 'b'], 'method': 'cosine@5', 'scores': [[4., 14.]], 'session_ids': ['s']}
    monkeypatch.setattr(api.pointer, 'score', original)
    result = asyncio.run(api.score(api.Request(sequences=[[[0.] * 128] * 2], session_ids=['s'])))
    assert result['scores'] == [[1., 2.]]
    assert result['accepted_at_targets']['0.001'] == [[False, False]]
    assert result['accepted'] == [[True, True]]
    assert result['training_selected_method'] == 'cosine@5'
    assert not result['candidate_promoted']
