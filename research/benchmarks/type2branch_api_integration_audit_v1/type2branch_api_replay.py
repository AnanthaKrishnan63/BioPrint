"""Actual frozen DEV retrieval and inference through in-process research ASGI API."""
import asyncio
import json
from pathlib import Path
import numpy as np
import httpx
from type2branch_continuity import ROOT, sha
from type2branch_calibrate import metrics
from type2branch_scoring import accept_scores

OUT=ROOT/'research/benchmarks/type2branch_api_replay_v1'


async def replay():
    from research_api import app
    import research_type2branch_api as api
    from type2branch_release import RELEASE, verify_release
    if api.MANIFEST_SHA256 is None:raise ValueError('API release pin not set')
    manifest=verify_release(api.MANIFEST_SHA256)
    dev=ROOT/'research/benchmarks/type2branch_dev_v1'
    report=json.loads((dev/'report.json').read_text())
    if sha(dev/'dev_scores.npz')!=report['output_sha256']:raise ValueError('Offline scores changed')
    with np.load(dev/'dev_scores.npz',allow_pickle=False) as a:
        expected=a['scores'];labels=a['labels'];subjects=a['subjects'];threshold=float(a['threshold'])
    if subjects.tolist()!=manifest['subjects'] or threshold!=manifest['threshold']:
        raise ValueError('Release/offline identity or threshold mismatch')
    OUT.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'code/bioprint/research_type2branch_api.py',
             ROOT/'code/bioprint/type2branch_worker_client.py',ROOT/'code/bioprint/research_api.py']
    (OUT/'plan.json').write_text(json.dumps({'scope':'All790DEV probes via API GET then actualworkerPOST',
        'manifest_sha256':api.MANIFEST_SHA256,'offline_scores_sha256':report['output_sha256'],
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'max_score_absolute_error':1e-5,'required_decision_flips':0},indent=2))
    scores=[];decisions=[];gate_checks={}
    transport=httpx.ASGITransport(app=app,client=('127.0.0.1',4321))
    try:
        async with httpx.AsyncClient(transport=transport,base_url='http://127.0.0.1',timeout=90) as client:
            for split in ['test','train']:
                response=await client.get(f'/api/type2branch/{split}/samples')
                gate_checks[split]=response.status_code;assert response.status_code==403
            response=await client.post('/api/type2branch/score',json={'features':[]})
            gate_checks['malformed']=response.status_code;assert response.status_code==422
            for header,value in [('Origin','http://remote.invalid'),('Host','remote.invalid')]:
                response=await client.get('/api/type2branch/results',headers={header:value})
                gate_checks[header]=response.status_code;assert response.status_code==403
            for offset in range(0,790,32):
                response=await client.get('/api/type2branch/dev/samples',params={'offset':offset,'limit':32})
                response.raise_for_status();batch=response.json()
                assert batch['accounts']==subjects.tolist() and batch['total']==790
                n=len(batch['features']);assert n==min(32,790-offset)
                assert batch['subjects']==subjects[labels[offset:offset+n]].tolist()
                response=await client.post('/api/type2branch/score',json={'features':batch['features']})
                response.raise_for_status();result=response.json()
                assert result['accounts']==subjects.tolist() and result['threshold']==threshold
                scores.extend(result['scores']);decisions.extend(result['accepted'])
                print(json.dumps({'replayed':offset+n,'total':790}),flush=True)
            response=await client.get('/api/type2branch/results');response.raise_for_status()
            assert response.json()==report
        remote=httpx.ASGITransport(app=app,client=('192.0.2.1',4321))
        async with httpx.AsyncClient(transport=remote,base_url='http://127.0.0.1') as client:
            response=await client.get('/api/type2branch/results')
            gate_checks['remote_client']=response.status_code;assert response.status_code==403
    finally:
        api.close_worker()
    scores=np.asarray(scores,dtype=np.float64);decisions=np.asarray(decisions,dtype=bool)
    assert scores.shape==expected.shape==(790,79)
    error=float(np.max(np.abs(scores-expected)))
    flips=int(np.count_nonzero(decisions!=accept_scores(expected,threshold)))
    assert error<=1e-5 and flips==0
    np.testing.assert_array_equal(decisions,accept_scores(scores,threshold))
    mask=np.arange(79)[None,:]!=labels[:,None]
    scored={'scores':scores,'labels':labels,'subjects':subjects,
            'genuine':scores[np.arange(790),labels],'impostor':scores[mask]}
    computed=metrics(scored,threshold)
    for key in ['far','frr','false_acceptances','false_rejections']:
        assert computed['pooled'][key]==report['metrics']['pooled'][key]
    np.savez_compressed(OUT/'api_scores.npz',scores=scores,accepted=decisions)
    result={'status':'actual_dev_api_replay_complete','probes':790,'claims':scores.size,
            'max_absolute_score_error':error,'decision_flips':flips,'gate_checks':gate_checks,
            'metrics':computed,'output_sha256':sha(OUT/'api_scores.npz'),
            'limitations':report['limitations']+['In-process ASGI; no network listener launched']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','limitations']}))


if __name__=='__main__':asyncio.run(replay())
