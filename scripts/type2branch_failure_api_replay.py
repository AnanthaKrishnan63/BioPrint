"""Replay the actual infeasible outcome through ASGI; no biometric predictions."""
import asyncio
import json
from pathlib import Path
import httpx
from type2branch_continuity import ROOT, sha


async def main():
    from research_api import app
    import research_type2branch_api as api
    path=ROOT/'research/benchmarks/type2branch_dev_features_v1/report.json'
    if sha(path)!=api.FAILURE_REPORT_SHA256:raise ValueError('Failure pin mismatch')
    failure=json.loads(path.read_text())
    out=ROOT/'research/benchmarks/type2branch_failure_api_v1';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    checks={}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('127.0.0.1',1234)),base_url='http://127.0.0.1') as client:
        response=await client.get('/api/type2branch/results');checks['results']=response.status_code
        assert response.status_code==200 and response.json()['validation']==failure
        assert not response.json()['scoring_available'] and failure['metrics']=={}
        for split,expected in [('dev',409),('train',403),('test',403)]:
            response=await client.get(f'/api/type2branch/{split}/samples');checks[split]=response.status_code
            assert response.status_code==expected
        response=await client.post('/api/type2branch/score',json={});checks['score']=response.status_code
        assert response.status_code==409
        for header in ['Host','Origin']:
            value='remote.invalid' if header=='Host' else 'http://remote.invalid'
            response=await client.get('/api/type2branch/results',headers={header:value})
            checks[header]=response.status_code;assert response.status_code==403
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('192.0.2.1',1234)),base_url='http://127.0.0.1') as client:
        response=await client.get('/api/type2branch/results');checks['remote_client']=response.status_code
        assert response.status_code==403
    assert api._worker is None
    report={'status':'actual_failure_api_replay_complete','checks':checks,'biometric_predictions':0,
            'failure_sha256':sha(path),'source_sha256':sha(Path(__file__)),
            'limitations':['Status/API gate replay only; no DEV scores or recognition metrics','No network listener launched']}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':asyncio.run(main())
