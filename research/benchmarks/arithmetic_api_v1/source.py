"""Actual in-process replay of frozen arithmetic status; never score a subset."""
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import research_api
import research_arithmetic_api as arithmetic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/arithmetic_api_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def replay():
    OUT.mkdir(exist_ok=False)
    plan = {'source_sha256': {name: sha(ROOT / name) for name in
        ['scripts/arithmetic_api_replay.py', 'code/bioprint/research_arithmetic_api.py',
         'code/bioprint/research_api.py']}, 'artifact_sha256': arithmetic.PINS,
        'scope': 'Status/error replay only; no biometric predictions or observations'}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    transport = httpx.ASGITransport(app=research_api.app, client=('127.0.0.1', 12345))
    statuses = {}
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        response = await client.get('/api/arithmetic/results')
        assert response.status_code == 200
        result = response.json()
        train, failure = arithmetic.artifacts()
        assert result['training_calibration'] == train['calibration']
        assert result['training_selection'] == train['selection']
        assert result['validation'] == failure and result['validation']['metrics'] == {}
        assert not result['validation_metrics_available'] and not result['scoring_available']
        assert response.headers['cache-control'] == 'no-store'
        statuses['results'] = response.status_code
        for split, expected in [('dev', 409), ('test', 403), ('train', 403)]:
            response = await client.get(f'/api/arithmetic/{split}/samples')
            assert response.status_code == expected
            statuses[split + '_samples'] = response.status_code
        response = await client.post('/api/arithmetic/score', json={})
        assert response.status_code == 409
        statuses['score'] = response.status_code
        response = await client.get('/api/arithmetic/results', headers={'origin': 'https://example.org'})
        assert response.status_code == 403
        statuses['cross_origin'] = response.status_code
    remote = httpx.ASGITransport(app=research_api.app, client=('192.0.2.1', 12345))
    async with httpx.AsyncClient(transport=remote, base_url='http://127.0.0.1') as client:
        response = await client.get('/api/arithmetic/results')
        assert response.status_code == 403
        statuses['remote'] = response.status_code
    report = dict(status='complete_status_replay', statuses=statuses,
                  frozen_reports_exact=True, biometric_scores_replayed=0,
                  validation_status='infeasible', listener_started=False, test_accessed=False,
                  plan_sha256=sha(OUT / 'plan.json'))
    (OUT / 'replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    asyncio.run(replay())
