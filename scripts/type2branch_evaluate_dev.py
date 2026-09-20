"""Frozen cross-session DEV inference; no parameter or threshold selection."""
import json
import os
from pathlib import Path
import sys
if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.research-type2branch-deps'))
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module
from type2branch_calibrate import metrics
from type2branch_cross_session import cross_session_scores
from type2branch_scoring import accept_scores
from keystroke_benchmark import curve_metrics

OUT=ROOT/'research/benchmarks/type2branch_dev_v1'


def cohort_results(scored, cohorts, threshold):
    output={}
    for name,identities in cohorts.items():
        columns=np.flatnonzero(np.isin(scored['subjects'],identities))
        if len(columns)!=len(identities):raise ValueError('Cohort membership mismatch')
        rows=np.flatnonzero(np.isin(scored['labels'],columns))
        labels=np.searchsorted(columns,scored['labels'][rows])
        matrix=scored['scores'][rows][:,columns]
        genuine=matrix[np.arange(len(rows)),labels]
        mask=np.arange(len(columns))[None,:]!=labels[:,None]
        local={'scores':matrix,'labels':labels,'subjects':scored['subjects'][columns],
               'genuine':genuine,'impostor':matrix[mask]}
        full=scored['scores'][rows]
        fullmask=np.arange(full.shape[1])[None,:]!=scored['labels'][rows,None]
        impostor=full[fullmask]
        output[name]={'within_cohort_claims':metrics(local,threshold),
            'full_account_claims_grouped_by_probe_identity':{
                'genuine_n':len(genuine),'impostor_n':len(impostor),
                'frr':float((~accept_scores(genuine,threshold)).mean()),
                'far':float(accept_scores(impostor,threshold).mean()),
                'discrete_eer':curve_metrics(genuine,impostor)['eer']}}
    return output


def main():
    features_dir=ROOT/'research/benchmarks/type2branch_dev_features_v1'
    prepared=json.loads((features_dir/'report.json').read_text())
    if prepared['status']!='frozen_dev_features_complete' or prepared['failures']:
        raise ValueError('Complete prescribed DEV cohort required')
    feature_plan=json.loads((features_dir/'plan.json').read_text())
    calibration=ROOT/'research/benchmarks/type2branch_calibration_v1'
    cal=json.loads((calibration/'report.json').read_text())
    cal_plan=json.loads((calibration/'plan.json').read_text())
    if cal['status']!='inner_train_calibration_complete' or cal_plan['checkpoint']!=feature_plan['checkpoint']:
        raise ValueError('Frozen calibration/model mismatch')
    if cal['threshold']!=feature_plan['threshold']:raise ValueError('Frozen threshold changed')
    threshold=float(cal['threshold'])
    if not np.isfinite(threshold):raise ValueError('Nonfinite threshold')
    checkpoint=Path(cal_plan['checkpoint']['checkpoint'])
    for filename,expected in cal_plan['checkpoint_sha256'].items():
        if sha(checkpoint.parent/filename)!=expected:raise ValueError('Checkpoint changed')
    protocol=feature_plan['protocol']
    for manifest in [protocol,feature_plan,cal_plan]:
        for name,expected in manifest['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError(f'Source changed:{name}')
    for role in ['gallery','probes']:
        if sha(features_dir/f'{role}.npz')!=prepared['roles'][role]['sha256']:
            raise ValueError('DEV feature artifact changed')
    OUT.mkdir(exist_ok=False)
    plan={'checkpoint':cal_plan['checkpoint'],'threshold':threshold,
          'feature_sha256':{role:prepared['roles'][role]['sha256'] for role in ['gallery','probes']},
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
              ROOT/'scripts/type2branch_calibrate.py',ROOT/'scripts/type2branch_cross_session.py',
              features_dir/'plan.json',features_dir/'report.json',calibration/'report.json',calibration/'plan.json']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__!='1.26.4':raise ValueError('Pinned NumPy required')
    ref=ROOT/'research/benchmarks/references/type2branch'
    module('conf',ref/'conf.small.1Kusers.py')
    model=module('type2branch_dev_model',ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    status=tf.train.Checkpoint(model=model).restore(str(checkpoint))
    status.assert_existing_objects_matched();status.expect_partial()
    arrays={}
    for role in ['gallery','probes']:
        with np.load(features_dir/f'{role}.npz',allow_pickle=False) as a:
            x=a['features'].astype('float32');subjects=a['subject'];windows=a['window']
            if not np.isfinite(x).all() or not (a['true_length']==100).all():raise ValueError('Invalid features')
        embedded=np.concatenate([model(x[start:start+32],training=False).numpy() for start in range(0,len(x),32)])
        arrays[role]=(embedded,subjects,windows)
    scored=cross_session_scores(*arrays['gallery'],*arrays['probes'],protocol['allowed_identities'])
    result=metrics(scored,threshold)
    assert result['pooled']['genuine_n']==790 and result['pooled']['impostor_n']==61620
    cohorts=cohort_results(scored,protocol['cohorts'],threshold)
    np.savez_compressed(OUT/'dev_scores.npz',**scored,threshold=np.float64(threshold),
        gallery_embeddings=arrays['gallery'][0],gallery_subjects=arrays['gallery'][1],gallery_windows=arrays['gallery'][2],
        probe_embeddings=arrays['probes'][0],probe_subjects=arrays['probes'][1],probe_windows=arrays['probes'][2])
    report={'status':'frozen_cross_session_dev_complete','metrics':result,'cohorts':cohorts,
            'threshold':threshold,'selected_epoch':cal_plan['checkpoint']['epoch'],
            'output_sha256':sha(OUT/'dev_scores.npz'),'limitations':protocol['limitations']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'pooled':result['pooled'],'macro':result['macro']}))


if __name__=='__main__':main()
