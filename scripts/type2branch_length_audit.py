"""TRAIN-selection-only feature truncation sensitivity; no new scoring policy."""
import json
import os
from pathlib import Path
import sys
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module

OUT=ROOT/'research/benchmarks/type2branch_length_audit_v1'


def truncate_features(features, length):
    import numpy as np
    x=np.asarray(features)
    if x.ndim!=3 or x.shape[1:]!=(100,5) or not np.isfinite(x).all():
        raise ValueError('Finite batch of100x5features required')
    if type(length) is not int or not 1<=length<=100:raise ValueError('Invalid prefix length')
    result=x.copy();result[:,length:,:]=0
    return result


def main():
    run=ROOT/'research/benchmarks/type2branch_train_v1'
    report=json.loads((run/'report.json').read_text())
    training_plan=json.loads((run/'plan.json').read_text())
    if report['status']!='initial_budget_complete':raise ValueError('Completed training required')
    selected=report['best_checkpoint'];checkpoint=Path(selected['checkpoint'])
    receipts=json.loads((run/'checkpoint_receipts_complete.json').read_text())
    for name,expected in receipts[str(selected['epoch'])]['files'].items():
        if sha(run/name)!=expected:raise ValueError('Checkpoint changed')
    inner=ROOT/'research/benchmarks/type2branch_inner_train_features_v1'
    features=inner/'selection.npz'
    expected=training_plan['input_sha256'][str(features.relative_to(ROOT))]
    if sha(features)!=expected:raise ValueError('Selection input changed')
    for name,digest in training_plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Training source changed')
    OUT.mkdir(exist_ok=False)
    plan={'scope':'Only16innerTRAINselection identities; no calibration/DEV/test arrays loaded',
          'lengths':[25,50,75,100],'perturbation':'Keep firstLstored feature rows; zero all5channels afterL',
          'gallery':'First5full100-event windows per identity, unchanged across lengths',
          'probes':'Last10windows, separately truncated in each diagnostic',
          'checkpoint':selected,'checkpoint_sha256':receipts[str(selected['epoch'])]['files'],
          'input_sha256':expected,'source_sha256':sha(Path(__file__)),
          'decision':'No threshold or minimumlength selected; diagnostic only',
          'limitations':['Triggered by disclosed DEV count infeasibility, but no DEV timings/scores used here',
            'Feature truncation is not exact shortened raw synthesis because fallback draw order can differ',
            'Model trained only on full100-event sequences; padding robustness unproven',
            'Selection identities already influenced checkpoint choice']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2));(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import numpy as np
    import tensorflow as tf
    from type2branch_scoring import gallery_scores
    from keystroke_benchmark import curve_metrics
    if np.__version__!='1.26.4':raise ValueError('Pinned NumPy required')
    ref=ROOT/'research/benchmarks/references/type2branch'
    module('conf',ref/'conf.small.1Kusers.py')
    model=module('type2branch_length_model',ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    status=tf.train.Checkpoint(model=model).restore(str(checkpoint))
    status.assert_existing_objects_matched();status.expect_partial()
    with np.load(features,allow_pickle=False) as a:
        ids=np.unique(a['subject']);grouped=[]
        for identity in ids:
            rows=np.flatnonzero(a['subject']==identity);rows=rows[np.argsort(a['window'][rows])]
            if not np.array_equal(a['window'][rows],np.arange(15)):raise ValueError('Window allocation changed')
            grouped.append(a['features'][rows].astype('float32'))
    x=np.stack(grouped);assert x.shape==(16,15,100,5)
    def embed(values):
        return np.concatenate([model(values[i:i+32],training=False).numpy() for i in range(0,len(values),32)])
    gallery=embed(x[:,:5].reshape(-1,100,5)).reshape(16,5,256)
    query=x[:,5:].reshape(-1,100,5);labels=np.repeat(np.arange(16),10)
    impostor_mask=np.arange(16)[None,:]!=labels[:,None]
    outputs={};arrays={}
    for length in plan['lengths']:
        embeddings=embed(truncate_features(query,length))
        scores=gallery_scores(embeddings,gallery)
        genuine=scores[np.arange(len(scores)),labels];impostor=scores[impostor_mask]
        outputs[str(length)]={'pooled_discrete_eer':curve_metrics(genuine,impostor)['eer'],
            'macro_discrete_eer':float(np.mean([curve_metrics(scores[labels==k,k],scores[labels!=k,k])['eer'] for k in range(16)])),
            'genuine_mean_distance':float(-genuine.mean()),'impostor_mean_distance':float(-impostor.mean()),
            'mean_embedding_norm':float(np.linalg.norm(embeddings,axis=1).mean()),
            'genuine_n':len(genuine),'impostor_n':len(impostor)}
        arrays[f'scores_{length}']=scores
        print(json.dumps({'length':length,**outputs[str(length)]}),flush=True)
    np.savez_compressed(OUT/'training_length_scores.npz',labels=labels,subjects=ids,**arrays)
    result={'status':'inner_train_length_sensitivity_complete','results':outputs,
            'output_sha256':sha(OUT/'training_length_scores.npz'),'limitations':plan['limitations']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
