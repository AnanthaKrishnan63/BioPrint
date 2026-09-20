"""Replay every eligible frozen DEV capture through API and actual encoder IPC."""
import asyncio
import json
from pathlib import Path
import httpx
import numpy as np
from type2branch_continuity import ROOT,sha
from type2branch_capture_metrics import capture_metrics
from type2branch_calibrate_lengths import verify_hashes

OUT=ROOT/'research/benchmarks/type2branch_capture_api_replay_v1'


async def replay():
    from research_api import app
    import research_type2branch_capture_api as api
    from type2branch_capture_release import verify_release
    if api.MANIFEST_SHA256 is None:raise ValueError('Capture API release pin not set')
    manifest=verify_release(api.MANIFEST_SHA256)
    dev=ROOT/'research/benchmarks/type2branch_capture_dev_evaluation_v1'
    report=json.loads((dev/'report.json').read_text())
    plan=json.loads((dev/'plan.json').read_text());verify_hashes(plan)
    if sha(dev/'scores.npz')!=report['scores_sha256']:raise ValueError('Offline scores changed')
    with np.load(dev/'scores.npz',allow_pickle=False) as data:
        expected=data['scores'];expected_decisions=data['decisions'];subjects=data['subjects']
        probe_subjects=data['probe_subjects'];lengths=data['probe_lengths']
    if subjects.tolist()!=manifest['subjects'] or expected.shape!=(len(probe_subjects),79):
        raise ValueError('Release/offline identity mismatch')
    thresholds={int(k):float(v) for k,v in manifest['thresholds'].items()}
    if thresholds!={int(k):v for k,v in plan['thresholds'].items()}:raise ValueError('Threshold binding mismatch')
    preparation=json.loads((ROOT/'research/benchmarks/type2branch_capture_dev_features_v1/report.json').read_text())
    OUT.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'code/bioprint/research_type2branch_capture_api.py',
        ROOT/'code/bioprint/type2branch_capture_worker_client.py',ROOT/'code/bioprint/type2branch_worker_client.py',
        ROOT/'code/bioprint/research_api.py']
    (OUT/'plan.json').write_text(json.dumps({'scope':'Every eligible DEV capture: GET then actual encoder-worker POST',
        'manifest_sha256':api.MANIFEST_SHA256,'expected_probes':len(probe_subjects),
        'offline_scores_sha256':report['scores_sha256'],'score_tolerance':1e-5,'decision_flips_required':0,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}},indent=2)+'\n')
    scores=[];decisions=[];checks={};total=len(probe_subjects)
    try:
        transport=httpx.ASGITransport(app=app,client=('127.0.0.1',4321))
        async with httpx.AsyncClient(transport=transport,base_url='http://127.0.0.1',timeout=90) as client:
            for split in ['train','test']:
                response=await client.get(f'/api/type2branch-capture/{split}/samples')
                checks[split]=response.status_code;assert response.status_code==403
            response=await client.post('/api/type2branch-capture/score',json={'features':[],'true_lengths':[]})
            checks['malformed']=response.status_code;assert response.status_code==422
            for key,value in [('Host','remote.invalid'),('Origin','http://remote.invalid')]:
                response=await client.get('/api/type2branch-capture/results',headers={key:value})
                checks[key]=response.status_code;assert response.status_code==403
            for offset in range(0,max(1,total),32):
                response=await client.get('/api/type2branch-capture/dev/samples',params={'offset':offset,'limit':32})
                response.raise_for_status();batch=response.json();count=min(32,total-offset)
                assert batch['accounts']==subjects.tolist() and batch['total']==total
                assert batch['subjects']==probe_subjects[offset:offset+count].tolist()
                assert batch['true_lengths']==lengths[offset:offset+count].tolist()
                assert len(batch['features'])==count
                if not count:continue
                response=await client.post('/api/type2branch-capture/score',json={
                    'features':batch['features'],'true_lengths':batch['true_lengths']})
                response.raise_for_status();result=response.json()
                assert result['accounts']==subjects.tolist() and result['thresholds']==manifest['thresholds']
                scores.extend(result['scores']);decisions.extend(result['accepted'])
                print(json.dumps({'replayed':offset+count,'total':total}),flush=True)
            response=await client.get('/api/type2branch-capture/results');response.raise_for_status()
            assert response.json()==report
        remote=httpx.ASGITransport(app=app,client=('192.0.2.1',4321))
        async with httpx.AsyncClient(transport=remote,base_url='http://127.0.0.1') as client:
            response=await client.get('/api/type2branch-capture/results')
            checks['remote_client']=response.status_code;assert response.status_code==403
    finally:api.close_worker()
    scores=np.asarray(scores,dtype=np.float64).reshape(total,79)
    decisions=np.asarray(decisions,dtype=bool).reshape(total,79)
    error=float(np.max(np.abs(scores-expected))) if total else 0.
    flips=int(np.count_nonzero(decisions!=expected_decisions))
    assert error<=1e-5 and flips==0
    row_thresholds=np.array([thresholds[int(n)] for n in lengths],dtype=np.float64)
    np.testing.assert_array_equal(decisions,scores>=row_thresholds[:,None])
    computed=capture_metrics(scores,probe_subjects,lengths,subjects,preparation['coverage'],
                             thresholds,plan['protocol']['cohorts'])
    assert computed['coverage']==report['metrics']['coverage']
    assert computed['whole_cohort']==report['metrics']['whole_cohort']
    if total:
        for key in ['far','frr','false_acceptances','false_rejections']:
            assert computed['conditional_metrics']['pooled'][key]==report['metrics']['conditional_metrics']['pooled'][key]
    np.savez_compressed(OUT/'api_scores.npz',scores=scores,accepted=decisions)
    result={'status':'actual_capture_dev_api_replay_complete','probes':total,'claims':scores.size,
        'max_absolute_score_error':error,'decision_flips':flips,'gate_checks':checks,'metrics':computed,
        'output_sha256':sha(OUT/'api_scores.npz'),
        'limitations':report['limitations']+['In-process ASGI with actual isolated model worker; no network listener']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','limitations']}))


if __name__=='__main__':asyncio.run(replay())
