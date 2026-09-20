"""Paired TRAIN-only full/mixed-prefix continuations; separate CPU processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import types
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module
from type2branch_loss_audit import definitions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['full','mixed'],required=True)
    args=parser.parse_args()
    shared=json.loads((ROOT/'research/benchmarks/type2branch_length_pair_v1/plan.json').read_text())
    for name,expected in shared['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Paired study source changed')
    out=ROOT/f'research/benchmarks/type2branch_length_train_{args.arm}_v1'
    run=ROOT/'research/benchmarks/type2branch_train_v1'
    original=json.loads((run/'report.json').read_text());oldplan=json.loads((run/'plan.json').read_text())
    selected=original['best_checkpoint']
    if original['status']!='initial_budget_complete' or selected['updates']!=300 or selected['epoch']!=2:
        raise ValueError('Expected frozen epoch2/300update starting point')
    checkpoint_path=Path(selected['checkpoint'])
    receipts=json.loads((run/'checkpoint_receipts_complete.json').read_text())['2']['files']
    for name,expected in receipts.items():
        if sha(run/name)!=expected:raise ValueError('Starting checkpoint changed')
    for name,expected in {**oldplan['input_sha256'],**oldplan['source_sha256']}.items():
        if sha(ROOT/name)!=expected:raise ValueError('Original training dependency changed')
    state_path=run/'epoch_002_random.json'
    fit=ROOT/'research/benchmarks/type2branch_fit_five_channels_v1/fit_features.npz'
    selection=ROOT/'research/benchmarks/type2branch_inner_train_features_v1/selection.npz'
    ref=ROOT/'research/benchmarks/references/type2branch'
    distance_path=ROOT/'research/benchmarks/references/type2branch_addons/metric_learning.py'
    files=[Path(__file__),ROOT/'scripts/type2branch_length_training_core.py',
        ROOT/'scripts/type2branch_scoring.py',ROOT/'scripts/keystroke_benchmark.py',state_path,
        run/'report.json',run/'plan.json',distance_path]
    out.mkdir(exist_ok=False)
    plan={'arm':args.arm,'scope':'Only47fit and16innerTRAINselection identities; no calibration/DEV/test reads',
        'starting_checkpoint':selected,'starting_checkpoint_sha256':receipts,
        'additional_epochs':[3,4,5],'steps_per_epoch':100,'additional_updates':300,'seed':20260921,
        'batch':'Unchanged author K10/N15 generator, dedicated restored Python batch RNG, cursor18of47',
        'augmentation':'full:100events; mixed: independent uniform prefix choice25/50/75/100per sample; zeroall5tailchannels',
        'selection':'Mean TRAINselection oracleFRRat1%FAR over four probe lengths, tie meanpooledEER, thenearliestupdate',
        'gallery':'First5full100windows perselectionID; last10probes; fixed allclaims',
        'checkpoint_selection':'Include warm-start checkpoint; compare both arms after both complete',
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [fit,selection]},
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'limitations':['Feature augmentation differs from shortened raw synthesis fallback ordering',
            'Dropout RNG starts fresh identically perarm; not exact continuation of original stochastic trajectory',
            'Epochs3..5 remain before source nearest-neighbor curriculum at20',
            'TRAINselection oracle thresholds are diagnostics, never deployed; calibration remains separate',
            'Study motivated by TRAIN length sensitivity following disclosedDEVmetadatafailure']}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import numpy as np
    import tensorflow as tf
    from type2branch_length_training_core import augment_prefixes, selection_key, tuple_state
    from type2branch_scoring import gallery_scores
    from keystroke_benchmark import curve_metrics
    tf.keras.utils.set_random_seed(plan['seed'])
    conf=module('conf',ref/'conf.small.1Kusers.py')
    assert conf.K==10 and conf.N==15 and conf.TRAINING_STEPS==100 and conf.CURRICULUM_DELAY==20
    built=module('type2branch_length_train_model',ref/'model.py').get_model_Type2Branch({'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})
    model,optimizer=built['model'],built['optimizer']
    optimizer.build(model.trainable_variables)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    restored=checkpoint.restore(str(checkpoint_path));restored.assert_consumed()
    assert int(optimizer.iterations.numpy())==300
    distance=definitions(distance_path,['pairwise_distance'],{'tf':tf,'TensorLike':tf.Tensor})
    scope=definitions(ref/'loss.py',['calculate_set2set_loss','Set2SetLoss'],
        {'tf':tf,'sys':sys,'Loss':tf.keras.losses.Loss,'conf':conf,
         'metric_learning':types.SimpleNamespace(pairwise_distance=distance['pairwise_distance'])})
    loss=scope['Set2SetLoss'](conf.K,conf.BETA)
    with np.load(fit,allow_pickle=False) as a:
        fitids=np.unique(a['subject'])
        x={str(k):{str(j):sample.astype('float32') for j,sample in enumerate(a['features'][a['subject']==k])} for k in fitids}
    assert len(x)==47 and all(len(v)==15 for v in x.values())
    with np.load(selection,allow_pickle=False) as a:
        ids=np.unique(a['subject']);groups=[]
        for identity in ids:
            rows=np.flatnonzero(a['subject']==identity);rows=rows[np.argsort(a['window'][rows])]
            assert np.array_equal(a['window'][rows],np.arange(15))
            groups.append(a['features'][rows].astype('float32'))
    sx=np.stack(groups);assert sx.shape==(16,15,100,5) and not set(ids)&set(fitids)
    gen_module=module('type2branch_length_generator',ref/'training_generator.py')
    batches,generator,_=gen_module.get_generator(built,x)
    # Advance only the iterator cursor, then reset its dedicated random state.
    for _ in range(300%47):next(batches)
    batch_rng=random.Random();batch_rng.setstate(tuple_state(json.loads(state_path.read_text())))
    augmentation_rng=np.random.default_rng(plan['seed'])
    batch_digest=hashlib.sha256()

    @tf.function
    def step(batch,target):
        with tf.GradientTape() as tape:value=loss(target,model(batch,training=True))
        gradients=tape.gradient(value,model.trainable_variables)
        tf.debugging.assert_all_finite(value,'Nonfinite loss')
        for g in gradients:
            if g is None:raise ValueError('Missing gradient')
            tf.debugging.assert_all_finite(tf.convert_to_tensor(g),'Nonfinite gradient')
        optimizer.apply_gradients(zip(gradients,model.trainable_variables));return value

    start=time.perf_counter()
    def emit(record):
        record.update(seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def embed(values):
        return np.concatenate([model(values[i:i+32],training=False).numpy() for i in range(0,len(values),32)])
    def evaluate(epoch):
        gallery=embed(sx[:,:5].reshape(-1,100,5)).reshape(16,5,256)
        queries=sx[:,5:].reshape(-1,100,5);labels=np.repeat(np.arange(16),10)
        impostors=np.arange(16)[None,:]!=labels[:,None];results={};saved={}
        for length in [25,50,75,100]:
            q=augment_prefixes(queries,np.full(len(queries),length,dtype=np.int64))
            scores=gallery_scores(embed(q),gallery)
            values=curve_metrics(scores[np.arange(len(scores)),labels],scores[impostors])
            results[str(length)]=values;saved[f'scores_{length}']=scores
        prefix=checkpoint.save(str(out/f'epoch_{epoch:03d}'))
        np.savez_compressed(out/f'epoch_{epoch:03d}_selection.npz',labels=labels,subjects=ids,**saved)
        record={'kind':'checkpoint','epoch':epoch,'updates':int(optimizer.iterations.numpy()),
            'checkpoint':prefix,'per_length':results,
            'mean_selection_frr_at_1pct':float(np.mean([v['oracle_frr_at_far_1pct'] for v in results.values()])),
            'mean_selection_eer':float(np.mean([v['eer'] for v in results.values()])),
            'batch_stream_sha256':batch_digest.hexdigest()}
        selection_key(record);emit(record)
        (out/f'epoch_{epoch:03d}_rng.json').write_text(json.dumps({'batch_python':batch_rng.getstate(),
            'augmentation':augmentation_rng.bit_generator.state,'generator_cursor':record['updates']%47}))
        return record
    records=[evaluate(2)]
    for epoch in plan['additional_epochs']:
        generator.on_epoch_begin(epoch);losses=[]
        for index in range(100):
            ambient=random.getstate();random.setstate(batch_rng.getstate())
            try:
                batch,target=next(batches);batch_rng.setstate(random.getstate())
            finally:random.setstate(ambient)
            assert batch.shape==(150,100,5) and target.shape==(150,)
            blocks=target.reshape(10,15);assert (blocks==blocks[:,:1]).all() and len(set(blocks[:,0]))==10
            batch_digest.update(batch.tobytes());batch_digest.update(target.tobytes())
            if args.arm=='mixed':
                lengths=augmentation_rng.choice([25,50,75,100],size=len(batch))
                batch=augment_prefixes(batch,lengths)
            value=float(step(batch,target).numpy());losses.append(value)
            if (index+1)%10==0:emit({'kind':'training','epoch':epoch,'step':index+1,
                'updates':int(optimizer.iterations.numpy()),'mean_loss':float(np.mean(losses[-10:]))})
        generator.on_epoch_end(epoch);records.append(evaluate(epoch))
    best=min(records,key=selection_key)
    report={'status':'paired_arm_complete','arm':args.arm,'updates':int(optimizer.iterations.numpy()),
        'best_checkpoint':best,'checkpoints':records,'batch_stream_sha256':batch_digest.hexdigest(),
        'seconds':time.perf_counter()-start,'limitations':plan['limitations']}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
