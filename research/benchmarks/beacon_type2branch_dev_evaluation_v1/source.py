"""Evaluate all frozen paired comparators on DEV without fitting or selection."""
import json
from pathlib import Path
import numpy as np
import joblib
from beacon_type2branch_train_fusion import attach_embeddings, pairs, measure
from beacon_type2branch_embed import ROOT, PROTOCOL, PROTOCOL_SHA256, verify_hashes
from type2branch_continuity import sha

PREP=ROOT/'research/benchmarks/beacon_type2branch_dev_features_v1'
EMBED=ROOT/'research/benchmarks/beacon_type2branch_dev_embeddings_v1'
TRAIN=ROOT/'research/benchmarks/beacon_type2branch_fusion_train_v1'
OUT=ROOT/'research/benchmarks/beacon_type2branch_dev_evaluation_v1'
FROZEN_SHA='fb612db18d0d2635506df53d470b06cc9b5e4115efdc8731293a44a37367a1fe'


def main():
    if sha(PROTOCOL)!=PROTOCOL_SHA256 or sha(TRAIN/'frozen.json')!=FROZEN_SHA:
        raise ValueError('Frozen protocol/calibration changed')
    protocol=json.loads(PROTOCOL.read_text());verify_hashes(protocol)
    frozen=json.loads((TRAIN/'frozen.json').read_text())
    if frozen['status']!='paired_train_fusion_complete' or set(frozen['models'])!=set(protocol['models']):
        raise ValueError('All frozen comparators required')
    prepared=json.loads((PREP/'report.json').read_text());embedded=json.loads((EMBED/'report.json').read_text())
    if prepared['status']!='paired_dev_features_complete' or embedded['status']!='paired_dev_embeddings_complete':
        raise ValueError('Complete DEV artifacts required')
    for path in [TRAIN,PREP,EMBED]:verify_hashes(json.loads((path/'plan.json').read_text()))
    if embedded['features_sha256']!=prepared['features_sha256']:raise ValueError('Embedding input mismatch')
    inputs={str((PREP/'paired_data.json').relative_to(ROOT)):prepared['paired_data_sha256'],
        str((EMBED/'embeddings.npz').relative_to(ROOT)):embedded['embeddings_sha256']}
    for name,c in frozen['models'].items():inputs[str((TRAIN/(name+'.joblib')).relative_to(ROOT))]=c['model_sha256']
    verify_hashes({'input_sha256':inputs})
    sources=[Path(__file__),PROTOCOL,TRAIN/'frozen.json',ROOT/'scripts/beacon_type2branch_train_fusion.py',
        ROOT/'scripts/beacon_type2branch_embed.py']+[p/n for p in [TRAIN,PREP,EMBED] for n in ['plan.json']]+[p/'report.json' for p in [PREP,EMBED]]
    plan={'protocol_sha256':PROTOCOL_SHA256,'frozen_sha256':FROZEN_SHA,
        'source_sha256':{**protocol['source_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in sources}},'input_sha256':inputs}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    data=json.loads((PREP/'paired_data.json').read_text())
    manifest=json.loads((ROOT/'datasets/beacon/split_manifest.json').read_text())
    expected={r['participant_id'] for r in manifest['files'] if r['split']=='dev'}
    if set(data)!=expected:raise ValueError('No successful-subset evaluation')
    lengths={(r['subject'],r['role'],float(w['start'])):w['true_length'] for r in prepared['type2branch_audit'] for w in r['windows']}
    with np.load(EMBED/'embeddings.npz',allow_pickle=False) as a:data=attach_embeddings(data,a,lengths)
    x,y,groups=pairs(data);np.savez_compressed(OUT/'dev_pairs.npz',X=x,y=y,groups=groups)
    results={};scores={}
    for name,c in frozen['models'].items():
        model=joblib.load(TRAIN/(name+'.joblib'));score=-model.decision_function(x[:,c['columns']])
        scores[name]=score
        results[name]={'pooled':measure(y,score,c['thresholds']),
            'by_probe_identity':{s:measure(y[groups[:,1]==s],score[groups[:,1]==s],c['thresholds']) for s in sorted(expected)}}
    np.savez_compressed(OUT/'scores.npz',labels=y,groups=groups,**scores)
    verify_hashes(plan)
    report={'status':'paired_dev_evaluation_complete','evaluation_split':'dev','subjects':sorted(expected),
        'claims':len(y),'results':results,'frozen_sha256':FROZEN_SHA,
        'pairs_sha256':sha(OUT/'dev_pairs.npz'),'scores_sha256':sha(OUT/'scores.npz'),
        'windows':prepared['windows'],'below25_windows':prepared['below25_windows'],
        'test_observations_decoded':False,'prior_dev_exposure':True,'model_updates':0,
        'limitations':protocol['limitations']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:r['pooled'] for name,r in results.items()}),flush=True)


if __name__=='__main__':main()
