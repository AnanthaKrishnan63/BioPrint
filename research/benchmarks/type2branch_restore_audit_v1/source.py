"""Generated-only restore/inference check using the saved initialization checkpoint."""
import json
import os
from pathlib import Path
import sys
import time
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module


def main():
    out=ROOT/'research/benchmarks/type2branch_restore_audit_v1'
    run=ROOT/'research/benchmarks/type2branch_train_v1'
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    saved=next(row for row in history if row['kind']=='checkpoint' and row['epoch']==-1)
    checkpoint=Path(saved['checkpoint'])
    assert checkpoint.parent==run and saved['updates']==0
    plan=json.loads((run/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Training source changed')
    pieces=list(run.glob(checkpoint.name+'.*'))
    out.mkdir(exist_ok=False)
    (out/'plan.json').write_text(json.dumps({'scope':'Generated inputs only; initialization checkpoint, not candidate selection',
        'checkpoint':str(checkpoint),'checkpoint_sha256':{p.name:sha(p) for p in pieces},
        'source_sha256':sha(Path(__file__))},indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import numpy as np
    import tensorflow as tf
    assert np.__version__=='1.26.4'
    ref=ROOT/'research/benchmarks/references/type2branch'
    module('conf',ref/'conf.small.1Kusers.py')
    model=module('type2branch_restore_model',ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    reader=tf.train.load_checkpoint(str(checkpoint))
    iteration_key='optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
    assert reader.has_tensor(iteration_key) and int(reader.get_tensor(iteration_key))==0
    restore=tf.train.Checkpoint(model=model)
    def reload():
        status=restore.restore(str(checkpoint))
        status.assert_existing_objects_matched()
        status.expect_partial()
    reload()
    rng=np.random.default_rng(984)
    x=rng.normal(0,.1,size=(2,100,5)).astype('float32')
    x[:,:,0]=rng.integers(0,256,size=(2,100))/255.
    x[:,:,1:3]=np.abs(x[:,:,1:3])
    start=time.perf_counter()
    expected=model(x,training=False).numpy()
    assert expected.shape==(2,256) and np.isfinite(expected).all()
    saved_variables=[v.numpy().copy() for v in model.variables]
    # Deliberately change every model variable, then prove restore reinstates all.
    for variable in model.variables:variable.assign(tf.zeros_like(variable))
    assert any(not np.array_equal(variable.numpy(),old) for variable,old in zip(model.variables,saved_variables))
    reload()
    for variable,old in zip(model.variables,saved_variables):np.testing.assert_array_equal(variable.numpy(),old)
    actual=model(x,training=False).numpy()
    np.testing.assert_array_equal(actual,expected)
    np.savez_compressed(out/'generated_inference.npz',inputs=x,embeddings=actual)
    report={'status':'generated_checkpoint_restore_pass','checkpoint_updates':0,
            'restored_model_variables':len(model.variables),'shape':list(actual.shape),
            'max_absolute_repeat_error':float(np.max(np.abs(actual-expected))),
            'seconds':time.perf_counter()-start,'numpy_version':np.__version__,
            'output_sha256':sha(out/'generated_inference.npz'),
            'limitations':['Initialization checkpoint only; trained checkpoint restore pending',
                           'No dataset observations, calibration or recognition metrics']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
