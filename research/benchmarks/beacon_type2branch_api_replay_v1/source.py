"""Replay every frozen paired DEV fusion decision through isolated ASGI."""
import asyncio
import json
from pathlib import Path
import numpy as np
import httpx
from beacon_type2branch_train_fusion import measure
from beacon_type2branch_embed import ROOT
from type2branch_continuity import sha

OUT=ROOT/'research/benchmarks/beacon_type2branch_api_replay_v1'
DEV=ROOT/'research/benchmarks/beacon_type2branch_dev_evaluation_v1'
TRAIN=ROOT/'research/benchmarks/beacon_type2branch_fusion_train_v1'


async def main():
    from research_api import app
    from research_beacon_type2branch_api import REPORT_SHA256, FROZEN_SHA256, BOUNDARY
    if sha(DEV/'report.json')!=REPORT_SHA256 or sha(TRAIN/'frozen.json')!=FROZEN_SHA256:
        raise ValueError('Frozen replay inputs changed')
    report=json.loads((DEV/'report.json').read_text());frozen=json.loads((TRAIN/'frozen.json').read_text())
    if sha(DEV/'scores.npz')!=report['scores_sha256']:raise ValueError('Offline scores changed')
    sources=[Path(__file__),ROOT/'code/bioprint/research_api.py',ROOT/'code/bioprint/research_beacon_type2branch_api.py',
        ROOT/'scripts/beacon_type2branch_train_fusion.py',DEV/'report.json',TRAIN/'frozen.json']
    plan={'boundary':BOUNDARY,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'input_sha256':{str((DEV/'scores.npz').relative_to(ROOT)):report['scores_sha256']}}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    with np.load(DEV/'scores.npz',allow_pickle=False) as a:offline={key:a[key] for key in a.files}
    prefix='/api/type2branch-beacon';results={}
    transport=httpx.ASGITransport(app=app,client=('127.0.0.1',12345))
    async with httpx.AsyncClient(transport=transport,base_url='http://127.0.0.1') as client:
        for split in ['train','test']:
            assert (await client.get(prefix+'/pairs',params={'split':split})).status_code==403
        assert (await client.post(prefix+'/score',json={'model':'type2branch','features':[[0.]]})).status_code==422
        assert (await client.post(prefix+'/score',json={'model':'unknown','features':[[0.]*35]})).status_code==404
        returned=await client.get(prefix+'/results');returned.raise_for_status()
        assert returned.json()=={**report,'boundary':BOUNDARY}
        pages=[]
        for offset in range(0,report['claims'],128):
            response=await client.get(prefix+'/pairs',params={'offset':offset,'limit':128});response.raise_for_status()
            page=response.json();assert page['total']==report['claims'] and page['offset']==offset
            pages.append(page)
        y=np.asarray(sum([p['y'] for p in pages],[]));groups=np.asarray(sum([p['groups'] for p in pages],[]))
        assert np.array_equal(y,offline['labels']) and np.array_equal(groups,offline['groups'])
        for name,config in frozen['models'].items():
            scores=[];accepted={key:[] for key in config['thresholds']}
            for page in pages:
                response=await client.post(prefix+'/score',json={'model':name,'features':page['x']});response.raise_for_status()
                value=response.json();assert value['thresholds']==config['thresholds']
                assert value['boundary']==BOUNDARY and value['score_direction']=='larger_is_impostor'
                scores.extend(value['scores'])
                for key in accepted:accepted[key].extend(value['accepted'][key])
            scores=np.asarray(scores);error=float(np.max(abs(scores-offline[name])))
            assert error<=1e-12
            for key,threshold in config['thresholds'].items():assert np.array_equal(accepted[key],offline[name]<=threshold)
            measured={'pooled':measure(y,scores,config['thresholds']),
                'by_probe_identity':{s:measure(y[groups[:,1]==s],scores[groups[:,1]==s],config['thresholds']) for s in report['subjects']}}
            assert measured==report['results'][name]
            results[name]={'claims':len(y),'max_score_error':error,'decision_flips':0,'all_pooled_and_per_probe_metrics_match':True}
        for headers in [{'host':'example.org'},{'origin':'https://example.org'}]:
            assert (await client.get(prefix+'/info',headers=headers)).status_code==403
    remote=httpx.ASGITransport(app=app,client=('10.0.0.2',12345))
    async with httpx.AsyncClient(transport=remote,base_url='http://127.0.0.1') as client:
        assert (await client.get(prefix+'/info')).status_code==403
    for name,expected in {**plan['source_sha256'],**plan['input_sha256']}.items():
        if sha(ROOT/name)!=expected:raise ValueError('Replay dependency changed')
    result={'status':'paired_dev_api_replay_complete','mode':'in-process ASGI, no listener',
        'boundary':BOUNDARY,'models':results,'split_host_origin_remote_guards':403,
        'malformed_input_guard':422,'unknown_model_guard':404,'test_observations_decoded':False}
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':asyncio.run(main())
