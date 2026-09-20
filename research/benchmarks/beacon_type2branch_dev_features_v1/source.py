"""Prepare exact paired DEV windows under the frozen transfer protocol."""
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

OUT = ROOT / 'research/benchmarks/beacon_type2branch_dev_features_v1'
PROTOCOL = ROOT / 'research/benchmarks/beacon_type2branch_protocol_v1/protocol.json'
PROTOCOL_SHA = 'b7c3abc32abb7112975251b3f43e7d0130aefa4e03b2e1dee3ad0ecb0ff5883a'
FUSION = ROOT / 'research/benchmarks/beacon_type2branch_fusion_train_v1'


def guard_role(row, entry):
    if row['split'] != 'dev' or row['role'] not in ('enrollment', 'probe'):
        raise ValueError('Only prescribed DEV enrollment/probe records permitted')
    partition = 'train' if row['role'] == 'enrollment' else 'dev'
    role = 'personal_enrollment' if row['role'] == 'enrollment' else 'probe'
    if (entry['cohort'] != 'dev' or entry['storage_partition'] != 'dev'
            or entry['record_partition'] != partition or entry['role'] != role
            or entry['global_model_fit_allowed'] is not False):
        raise ValueError('DEV support/probe role mismatch or global fitting permission')


def verify_fusion(protocol):
    """Gate all observation access on the completed, fully frozen TRAIN result."""
    frozen_path = FUSION/'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if (frozen['status'] != 'paired_train_fusion_complete'
            or frozen['protocol_sha256'] != PROTOCOL_SHA
            or frozen['protocol'] != protocol or len(frozen['models']) != 8
            or set(frozen['models']) != set(protocol['models'])
            or frozen['identity_groups'] != protocol['identity_groups']):
        raise ValueError('Complete protocol-bound TRAIN fusion required before DEV')
    plan = json.loads((FUSION/'plan.json').read_text())
    if plan['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Fusion plan protocol changed')
    for group in ['source_sha256','input_sha256']:
        for name, expected in plan[group].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Fusion dependency changed: '+name)
    for name, config in frozen['models'].items():
        if (config['columns'] != protocol['models'][name]
                or config['C'] not in protocol['candidate_C']
                or set(config['thresholds']) != {'far_1pct','far_5pct','eer'}
                or not all(np.isfinite(v) for v in config['thresholds'].values())
                or digest(FUSION/f'{name}.joblib') != config['model_sha256']
                or digest(FUSION/f'{name}_train_scores.npz') != config['train_scores_sha256']):
            raise ValueError('Fusion model/threshold/score freeze mismatch')
    for name, expected in frozen['pairs_sha256'].items():
        if digest(FUSION/f'{name}_pairs.npz') != expected:
            raise ValueError('Frozen TRAIN pairs changed')
    return frozen_path


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    manifest, ledger = dual.base.manifests()
    frozen_path = verify_fusion(protocol)
    frozen_sha = digest(frozen_path)
    files = [r for r in manifest['files'] if r['split'] == 'dev']
    expected_subjects = {r['participant_id'] for r in files}
    if not expected_subjects or expected_subjects & set(sum(protocol['identity_groups'], [])):
        raise ValueError('Disjoint nonempty DEV cohort required')
    receipts_path = ROOT/'datasets/beacon/download_receipts.json'
    receipts = {r['path']:r for r in json.loads(receipts_path.read_text())}
    tree_path = ROOT/'datasets/beacon/source_tree_metadata.json'
    tree = {r['path']:r for r in json.loads(tree_path.read_text())}
    inputs = {}
    for row in files:
        entry = ledger[row['release_path']]
        guard_role(row, entry)
        path = dual.base.DATA/'dev'/row['release_path']
        name = str(path.relative_to(ROOT)); receipt = receipts[name]
        if (receipt['split'] != 'dev' or receipt['partial'] != row['partial']
                or digest(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']):
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
    previous_path = ROOT/'research/benchmarks/beacon-neural-v2/dev_results.json'
    sources = [Path(__file__), PROTOCOL, receipts_path, tree_path, previous_path, frozen_path, FUSION/'plan.json']
    sources += [FUSION/f'{name}.joblib' for name in protocol['models']]
    sources += [ROOT/'scripts'/name for name in ['beacon_type2branch_adapter.py',
        'type2branch_context_model.py', 'type2branch_synthesis_cleanup.py',
        'type2branch_residual_features.py', 'type2branch_csv_bridge.py']]
    plan = {'protocol_sha256':PROTOCOL_SHA, 'scope':'DEV personal enrollment and validation probes only; no global fitting',
        'fusion_frozen_sha256':frozen_sha,
        'source_sha256':{**protocol['source_sha256'], **{str(p.relative_to(ROOT)):digest(p) for p in sources}},
        'input_sha256':inputs}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    data, audits = dual.load('dev', dual.base.Encoder())
    if set(data) != expected_subjects:
        raise ValueError('Unexpected identity loss; no subset experiment')
    previous = json.loads(previous_path.read_text())
    if set(previous['subjects']) != expected_subjects:
        raise ValueError('Prior matched DEV cohort differs from manifest')
    original = {(r['subject'],r['role']):r['eligible_window_starts'] for r in previous['dev_audit']}
    if set(original) != {(s,r) for s in expected_subjects for r in ['enrollment','probe']}:
        raise ValueError('Prior DEV audit does not cover exact cohort/roles')
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
            parsed=parse_rows(dual.base.baseline.read_csv(dual.base.DATA/'dev'/row['release_path']))
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
    verify_fusion(protocol)
    report={'status':'paired_dev_features_complete','features_sha256':digest(OUT/'features.npz'),
        'paired_data_sha256':digest(OUT/'paired_data.json'),'subjects':sorted(data),
        'windows':len(features),'below25_windows':sum(n<25 for n in lengths),
        'true_length_counts':{str(n):lengths.count(n) for n in sorted(set(lengths))},
        'old_pipeline_audit':audits,'type2branch_audit':feature_audit,
        'evaluation_split':'dev','sealed_test_payloads_decoded':False,
        'fusion_frozen_sha256':frozen_sha,'protocol_sha256':PROTOCOL_SHA,'metrics':{}}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','subjects','windows','below25_windows']}),flush=True)


if __name__=='__main__':main()
