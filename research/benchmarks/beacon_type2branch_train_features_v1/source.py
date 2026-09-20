"""Prepare exact paired TRAIN windows under the frozen transfer protocol."""
import json
import hashlib
from pathlib import Path
import numpy as np
import beacon_dual_neural_benchmark as dual
from beacon_type2branch_protocol import ROOT, digest
from beacon_type2branch_adapter import parse_rows, adapt_window
from type2branch_variable_adapter import synthesize_window
from type2branch_context_model import MeanContextModel
from type2branch_random import first_synthesis_thread

OUT = ROOT / 'research/benchmarks/beacon_type2branch_train_features_v1'
PROTOCOL = ROOT / 'research/benchmarks/beacon_type2branch_protocol_v1/protocol.json'
PROTOCOL_SHA = 'b7c3abc32abb7112975251b3f43e7d0130aefa4e03b2e1dee3ad0ecb0ff5883a'


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    manifest, ledger = dual.base.manifests()
    expected_subjects = set(sum(protocol['identity_groups'], []))
    files = [r for r in manifest['files'] if r['split'] == 'train']
    if {r['participant_id'] for r in files} != expected_subjects:
        raise ValueError('TRAIN cohort mismatch')
    receipts_path = ROOT/'datasets/beacon/download_receipts.json'
    receipts = {r['path']:r for r in json.loads(receipts_path.read_text())}
    tree_path = ROOT/'datasets/beacon/source_tree_metadata.json'
    tree = {r['path']:r for r in json.loads(tree_path.read_text())}
    inputs = {}
    for row in files:
        entry = ledger[row['release_path']]
        if entry['cohort'] != 'train' or entry['record_partition'] != 'train' or not entry['global_model_fit_allowed']:
            raise ValueError('Non-TRAIN observation prohibited')
        path = dual.base.DATA/'train'/row['release_path']
        name = str(path.relative_to(ROOT)); receipt = receipts[name]
        if receipt['split'] != 'train' or digest(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
            raise ValueError('Acquisition receipt mismatch')
        if not row['partial']:
            source = tree[row['release_path']]
            payload = path.read_bytes()
            actual = (digest(path) if 'lfs' in source else
                hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest())
            expected = source['lfs']['oid'] if 'lfs' in source else source['oid']
            if actual != expected or not receipt['pinned_source_integrity_verified']:
                raise ValueError('Pinned source revision integrity mismatch')
        inputs[name] = receipt['sha256']
    sources = [Path(__file__), PROTOCOL, receipts_path, tree_path]
    sources += [ROOT/'scripts'/name for name in ['beacon_type2branch_adapter.py',
        'type2branch_context_model.py', 'type2branch_synthesis_cleanup.py',
        'type2branch_residual_features.py', 'type2branch_csv_bridge.py']]
    plan = {'protocol_sha256':PROTOCOL_SHA, 'scope':'TRAIN only, original fit/selection/calibration roles',
        'source_sha256':{**protocol['source_sha256'], **{str(p.relative_to(ROOT)):digest(p) for p in sources}},
        'input_sha256':inputs}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    data, audits = dual.load('train', dual.base.Encoder())
    if set(data) != expected_subjects:
        raise ValueError('Unexpected identity loss; no subset experiment')
    previous = json.loads((ROOT/'research/benchmarks/beacon-neural-v2/frozen.json').read_text())
    original = {(r['subject'],r['role']):r['eligible_window_starts'] for r in previous['train_audit']}
    population = MeanContextModel()
    with np.load(ROOT/'research/benchmarks/type2branch_context_fit_v1/population.npz',allow_pickle=False) as p:
        for f,o,k,n,m,s in zip(p['feature'],p['order'],p['hashes'],p['count'],p['mean'],p['mean_square'],strict=True):
            key=(int(f),int(o),int(k))
            if key in population.models:raise ValueError('Duplicate population context')
            population.models[key]=(int(n),float(m),float(s))
    rng=first_synthesis_thread(); features=[]; lengths=[]; subjects=[]; roles=[]; starts=[]; feature_audit=[]
    for subject in sorted(data):
        for role in ['enrollment','probe']:
            windows = data[subject][role]['windows']
            if [w['start'] for w in windows] != original[subject,role]:
                raise ValueError('Paired window allocation changed')
            row=next(r for r in files if r['participant_id']==subject and r['role']==role and r['modality']=='keyboard_csv')
            parsed=parse_rows(dual.base.baseline.read_csv(dual.base.DATA/'train'/row['release_path']))
            audit=parsed['audit']
            if audit['unknown_codes']/audit['events']>.01 or audit['median_duration_disagreement_ms']>10:
                raise ValueError('Frozen key/clock quality gate failed')
            wa=[]
            for window in windows:
                adapted, info=adapt_window(parsed,window['start'])
                result=synthesize_window(adapted,population,rng)
                features.append(result['features']);lengths.append(result['true_length'])
                subjects.append(subject);roles.append(role);starts.append(window['start']);wa.append(info)
            feature_audit.append({'subject':subject,'role':role,'recording':audit,'windows':wa})
    np.savez_compressed(OUT/'features.npz',features=np.asarray(features,dtype=np.float32),
        true_length=np.asarray(lengths),subject=np.asarray(subjects),role=np.asarray(roles),start=np.asarray(starts))
    # Save matched old representations; no models or recognition thresholds are fitted here.
    (OUT/'paired_data.json').write_text(json.dumps(data)+'\n')
    for name,expected in {**plan['source_sha256'],**inputs}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Input/source changed during preparation')
    report={'status':'paired_train_features_complete','features_sha256':digest(OUT/'features.npz'),
        'paired_data_sha256':digest(OUT/'paired_data.json'),'subjects':sorted(data),
        'windows':len(features),'below25_windows':sum(n<25 for n in lengths),
        'true_length_counts':{str(n):lengths.count(n) for n in sorted(set(lengths))},
        'old_pipeline_audit':audits,'type2branch_audit':feature_audit,
        'dev_test_observations_decoded':False,'metrics':{}}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','subjects','windows','below25_windows']}),flush=True)


if __name__=='__main__':main()
