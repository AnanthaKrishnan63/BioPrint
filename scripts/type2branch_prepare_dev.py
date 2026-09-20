"""Gated, frozen cross-session DEV feature preparation; no test payload parsing."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, DATA, sha
from type2branch_calibrate import completed_checkpoint
from type2branch_context_model import MeanContextModel
from type2branch_keyrecs_adapter import adapt, parse_event
from type2branch_random import first_synthesis_thread, fill_average_fallback
from type2branch_residual_features import residual_features
from type2branch_synthesis_cleanup import cleanup

OUT=ROOT/'research/benchmarks/type2branch_dev_features_v1'


def select_dev_rows(lines, allowed):
    if not next(lines).startswith('participant,session,key1,key2,'):
        raise ValueError('Unexpected header')
    grouped=defaultdict(list)
    for line in lines:
        subject,session,opaque=line.split(',',2)
        if subject in allowed and session in ('1','2'):
            grouped[(subject,session)].append(opaque)
    return grouped


def require_allocation(raw, base, windows):
    if not windows or min(windows)<0 or max(windows)>=len(base):
        raise ValueError('Prescribed windows unavailable; no replacement allowed')
    if len(set(windows))!=len(windows):raise ValueError('Duplicate prescribed windows')
    return raw[windows],base[windows]


def adapt_prescribed_prefix(rows, windows):
    """Validate only events needed for allocation; never reinterpret a cut as EOF."""
    required=(max(windows)+1)*100
    if len(rows)<required:raise ValueError('Prescribed event prefix unavailable')
    prefix=rows[:required]
    if len(rows)>required:
        parse_event(prefix[-1],last=False) # A cut point is not a true terminal record.
    raw,base,audit=adapt(prefix)
    raw,base=require_allocation(raw,base,windows)
    return raw,base,audit


def main():
    run=ROOT/'research/benchmarks/type2branch_train_v1'
    chosen=completed_checkpoint(json.loads((run/'report.json').read_text()),json.loads((run/'plan.json').read_text()))
    calibration=ROOT/'research/benchmarks/type2branch_calibration_v1'
    calibrated=json.loads((calibration/'report.json').read_text())
    calibration_plan=json.loads((calibration/'plan.json').read_text())
    if calibrated['status']!='inner_train_calibration_complete' or calibration_plan['checkpoint']!=chosen:
        raise ValueError('Completed calibration does not match selected checkpoint')
    if not np.isfinite(calibrated['threshold']):raise ValueError('Nonfinite calibration threshold')
    if sha(calibration/'calibration_scores.npz')!=calibrated['scores_sha256']:
        raise ValueError('Calibration score artifact changed')
    protocol_path=ROOT/'research/benchmarks/type2branch_dev_protocol_v1/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen DEV source changed')
    manifest_path=DATA/'split-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    allowed=set(protocol['allowed_identities'])
    if len(allowed)!=79 or allowed & set(manifest['sealed_test_subjects']) or allowed!=set(manifest['active_subjects']):
        raise ValueError('Invalid DEV identity cohort')
    groups=[set(group) for group in protocol['cohorts'].values()]
    if set.union(*groups)!=allowed or sum(map(len,groups))!=len(allowed):
        raise ValueError('Cohort partition is not disjoint and exhaustive')
    prior_path=ROOT/'research/benchmarks/type2branch_inner_train_features_v1/plan.json'
    prior=json.loads(prior_path.read_text())
    for name,expected in prior['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('InnerTRAIN feature dependency changed')
    population_dir=ROOT/'research/benchmarks/type2branch_context_fit_v1'
    population=population_dir/'population.npz'
    population_hash=json.loads((population_dir/'report.json').read_text())['population_sha256']
    if population_hash!=prior['input_sha256'][str(population.relative_to(ROOT))]:
        raise ValueError('Population differs from innerTRAIN synthesis population')
    source=DATA/'free-text.csv'
    inputs={source:manifest['files']['free-text.csv']['sha256'],population:population_hash}
    for path,expected in inputs.items():
        if sha(path)!=expected:raise ValueError('Input changed')
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),protocol_path,manifest_path,prior_path,calibration/'report.json',calibration/'plan.json',
           ROOT/'scripts/type2branch_synthesis_cleanup.py',ROOT/'scripts/typenet_benchmark.py']
    plan={'protocol':protocol,'threshold':calibrated['threshold'],'checkpoint':chosen,
          'parsing_scope':'Only required prefix: galleryS1first2000events; probeS2first1000events. Cut is not treated as session EOF; later timing payloads remain unparsed.',
          'input_sha256':{str(p.relative_to(ROOT)):h for p,h in inputs.items()},
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    model=MeanContextModel()
    with np.load(population,allow_pickle=False) as p:
        for f,order,key,count,mean,square in zip(p['feature'],p['order'],p['hashes'],p['count'],p['mean'],p['mean_square'],strict=True):
            identity=(int(f),int(order),int(key))
            if identity in model.models:raise ValueError('Duplicate population entry')
            model.models[identity]=(int(count),float(mean),float(square))
    with source.open() as stream:grouped=select_dev_rows(stream,allowed)
    results={};failures=[]
    for role in ['gallery','probes']:
        session=protocol[role]['session'];windows=protocol[role]['windows']
        rng=first_synthesis_thread()
        arrays=[];masks=[];subjects=[];indices=[];raws=[];synthetics=[];orders_all=[];audit={}
        for subject in sorted(allowed):
            try:
                raw,base,info=adapt_prescribed_prefix(grouped.get((subject,session),[]),windows)
                audit[subject]=info
                for index,wire,observed in zip(windows,raw,base,strict=True):
                    predicted,orders=model.predict(wire[:,0])
                    generated=np.column_stack((wire[:,0],fill_average_fallback(predicted,rng)))
                    cleaned,partitions=cleanup(generated)
                    if len(partitions):raise ValueError('Unexpected synthetic pause partition')
                    features,valid=residual_features(observed,cleaned)
                    arrays.append(features);masks.append(valid);subjects.append(subject);indices.append(index)
                    raws.append(wire);synthetics.append(cleaned);orders_all.append(orders)
            except (ValueError,IndexError,TypeError) as error:
                failures.append({'role':role,'subject':subject,'error':str(error)})
        if failures:continue
        arrays=np.stack(arrays);masks=np.stack(masks);orders_all=np.stack(orders_all)
        if arrays.shape!=(79*len(windows),100,5) or not np.isfinite(arrays).all():
            raise ValueError('Unexpected complete-cohort feature shape')
        path=OUT/f'{role}.npz'
        np.savez_compressed(path,features=arrays,residual_valid=masks,subject=np.array(subjects),
                            window=np.array(indices),true_length=np.full(len(arrays),100),
                            signed_raw_ms=np.stack(raws),synthetic_ms=np.stack(synthetics),context_orders=orders_all)
        results[role]={'shape':list(arrays.shape),'sha256':sha(path),
                       'fallback_HT_FT':(orders_all<0).sum(axis=(0,1)).tolist(),
                       'missing_residual_HT_FT':(~masks).sum(axis=(0,1)).tolist(),'per_identity':audit}
    report={'status':'infeasible_prescribed_cohort' if failures else 'frozen_dev_features_complete',
            'roles':results,'failures':failures,'metrics':{},'sealed_test_payloads_decoded':False,
            'limitations':protocol['limitations']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({**report,'roles':{k:{a:b for a,b in v.items() if a!='per_identity'} for k,v in results.items()}}))


if __name__=='__main__':main()
