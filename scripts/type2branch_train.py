"""Bounded actual-data training with unchanged author model/loss/generator.

Three complete source-length epochs are an initial CPU budget, not convergence.
Checkpoint comparison uses only fixed inner-TRAIN selection batches.
"""
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import types
from type2branch_continuity import ROOT, sha
from type2branch_loss_audit import definitions
from type2branch_reference_smoke import module

REF = ROOT/'research/benchmarks/references/type2branch'
OUT = ROOT/'research/benchmarks/type2branch_train_v1'


def main():
    OUT.mkdir(exist_ok=False)
    fit_dir = ROOT/'research/benchmarks/type2branch_fit_five_channels_v1'
    inner = ROOT/'research/benchmarks/type2branch_inner_train_features_v1'
    paths = {fit_dir/'fit_features.npz': json.loads((fit_dir/'report.json').read_text())['output_sha256'],
             inner/'selection.npz': json.loads((inner/'report.json').read_text())['roles']['selection']['sha256']}
    distance_path = ROOT/'research/benchmarks/references/type2branch_addons/metric_learning.py'
    sources = [Path(__file__), distance_path, ROOT/'scripts/type2branch_loss_audit.py',
               ROOT/'scripts/type2branch_reference_smoke.py',
               ROOT/'.research-type2branch-deps/tf_keras/src/backend.py']
    sources += [REF/n for n in ['model.py','loss.py','training_generator.py','conf.small.1Kusers.py']]
    plan = {'scope':'47fit identities for gradients;16innerTRAINselection identities for checkpoint loss only',
            'epochs':3, 'steps_per_epoch':100, 'K':10, 'N':15, 'seed':20260920,
            'optimizer':'Author Adam1e-4', 'beta':0.05,
            'curriculum':'Unchanged source generator; delay20epochs, not reached in initial3epochs',
            'checkpoint_selection':'Lowest finite mean Set2Set loss on fixed two selection batches; earliest tie',
            'selection_batches':'Sorted IDs[0:10] and[6:16], all15windows; middle6 IDs appear twice',
            'scoring_contract':'Later first5gallery/last10probe, score to be frozen before calibration',
            'budget':'Initial300updates, checkpointed for continuation; no convergence claim',
            'adaptations':['Prepared Average/residual features as explicitly documented',
                           'Fixed selection batches instead of stochastic source validation',
                           'Save every epoch including initialization; no initial-loss<1 save gate',
                           'CPU TensorFlow2.16 overlay with recorded Keras Python3.12 compatibility patch'],
            'input_sha256':{str(p.relative_to(ROOT)):h for p,h in paths.items()},
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    for p in sources[:1]: (OUT/p.name).write_bytes(p.read_bytes())
    for p,h in paths.items():
        if sha(p)!=h: raise ValueError('Frozen input changed')
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',
        TF_NUM_INTEROP_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(plan['seed'])
    conf = module('conf',REF/'conf.small.1Kusers.py')
    assert conf.N==15 and conf.K==10 and conf.TRAINING_STEPS==100 and conf.CURRICULUM_DELAY==20
    built = module('type2branch_train_model',REF/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})
    model, optimizer = built['model'], built['optimizer']
    distance = definitions(distance_path,['pairwise_distance'],{'tf':tf,'TensorLike':tf.Tensor})
    scope = definitions(REF/'loss.py',['calculate_set2set_loss','Set2SetLoss'],
        {'tf':tf,'sys':sys,'Loss':tf.keras.losses.Loss,'conf':conf,
         'metric_learning':types.SimpleNamespace(pairwise_distance=distance['pairwise_distance'])})
    loss = scope['Set2SetLoss'](conf.K,conf.BETA)
    with np.load(fit_dir/'fit_features.npz',allow_pickle=False) as a:
        ids=np.unique(a['subject']); assert len(ids)==47
        x={str(identity):{str(j):sample.astype('float32') for j,sample in enumerate(a['features'][a['subject']==identity])} for identity in ids}
        assert all(len(v)==15 for v in x.values())
    with np.load(inner/'selection.npz',allow_pickle=False) as a:
        selection_ids=np.unique(a['subject']); assert len(selection_ids)==16 and not set(ids)&set(selection_ids)
        selection={identity:a['features'][a['subject']==identity].astype('float32') for identity in selection_ids}
        assert all(len(v)==15 for v in selection.values())
    selection_batches=[np.concatenate([selection[k] for k in group])
                       for group in [selection_ids[:10],selection_ids[6:16]]]
    labels=tf.repeat(tf.range(10),15)
    generator_mod=module('type2branch_train_generator',REF/'training_generator.py')
    batches,generator,callback=generator_mod.get_generator(built,x)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)

    @tf.function
    def step(batch, target):
        with tf.GradientTape() as tape:
            value=loss(target,model(batch,training=True))
        gradients=tape.gradient(value,model.trainable_variables)
        tf.debugging.assert_all_finite(value,'Nonfinite training loss')
        for g in gradients:
            if g is None: raise ValueError('Missing gradient')
            tf.debugging.assert_all_finite(tf.convert_to_tensor(g),'Nonfinite gradient')
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return value

    def emit(record):
        record['elapsed_seconds']=time.perf_counter()-started
        record['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def evaluate(epoch):
        values=[float(loss(labels,model(batch,training=False)).numpy()) for batch in selection_batches]
        if not np.isfinite(values).all(): raise ValueError('Nonfinite selection loss')
        prefix=checkpoint.save(str(OUT/f'epoch_{epoch:03d}'))
        (OUT/f'epoch_{epoch:03d}_random.json').write_text(json.dumps(random.getstate()))
        # Checkpoints preserve optimizer/model; Python state retained. Exact TF RNG resume unproven.
        result={'kind':'checkpoint','epoch':epoch,'updates':int(optimizer.iterations.numpy()),
                'selection_loss':float(np.mean(values)),'selection_batch_losses':values,'checkpoint':prefix}
        emit(result)
        return result

    started=time.perf_counter()
    records=[evaluate(-1)]
    for epoch in range(plan['epochs']):
        generator.on_epoch_begin(epoch)
        losses=[]
        for index in range(100):
            batch,target=next(batches)
            assert batch.shape==(150,100,5) and target.shape==(150,)
            blocks=target.reshape(10,15)
            assert all((block==block[0]).all() for block in blocks) and len(set(blocks[:,0]))==10
            value=float(step(batch,target).numpy()); losses.append(value)
            if (index+1)%10==0:
                emit({'kind':'training','epoch':epoch,'step':index+1,
                      'updates':int(optimizer.iterations.numpy()),'mean_loss':float(np.mean(losses[-10:]))})
        generator.on_epoch_end(epoch)
        records.append(evaluate(epoch))
    best=min(records,key=lambda r:r['selection_loss'])
    report={'status':'initial_budget_complete','optimizer_updates':int(optimizer.iterations.numpy()),
            'best_checkpoint':best,'checkpoints':records,'parameters':model.count_params(),
            'seconds':time.perf_counter()-started,
            'limitations':['Only3epochs, before source curriculum begins; continuation warranted',
                           'No calibration/DEV/test arrays loaded; no recognition metrics',
                           'Exact RNG-equivalent resumed trajectory not verified']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
