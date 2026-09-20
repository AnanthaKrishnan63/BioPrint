"""Calibrate a frozen completed training run; never consult DEV/test features."""
import json
import os
from pathlib import Path
import sys
# Executable inference must select NumPy1.26 before any NumPy-dependent import.
# Imported pure gate/metric helpers remain usable by the general test runner.
if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.research-type2branch-deps'))
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module
from type2branch_scoring import chronological_scores, global_threshold, accept_scores
from keystroke_benchmark import curve_metrics

RUN=ROOT/'research/benchmarks/type2branch_train_v1'
OUT=ROOT/'research/benchmarks/type2branch_calibration_v1'


def completed_checkpoint(report, plan):
    expected=plan['epochs']*plan['steps_per_epoch']
    if report.get('status')!='initial_budget_complete' or report.get('optimizer_updates')!=expected:
        raise ValueError('Training budget must be complete before calibration')
    records=report['checkpoints']
    if [r['epoch'] for r in records]!=list(range(-1,plan['epochs'])):
        raise ValueError('Missing or reordered epoch checkpoints')
    for record in records:
        if record['updates']!=(record['epoch']+1)*plan['steps_per_epoch']:
            raise ValueError('Checkpoint update count mismatch')
        if not np.isfinite(record['selection_loss']):raise ValueError('Nonfinite selection loss')
    chosen=min(records,key=lambda row:row['selection_loss'])
    if chosen!=report['best_checkpoint']:
        raise ValueError('Reported checkpoint violates frozen minimum-loss selection')
    return chosen


def metrics(scored, threshold):
    scores=np.asarray(scored['scores'],dtype=np.float64)
    labels=scored['labels'];subjects=scored['subjects']
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    rows=[]
    for index,subject in enumerate(subjects):
        genuine=scores[labels==index,index];impostor=scores[labels!=index,index]
        rows.append({'subject':str(subject),'genuine_n':len(genuine),'impostor_n':len(impostor),
                     'frr':float((~accept_scores(genuine,threshold)).mean()),
                     'far':float(accept_scores(impostor,threshold).mean()),
                     'discrete_eer':curve_metrics(genuine,impostor)['eer']})
    g,i=scored['genuine'],scored['impostor']
    return {'pooled':{'genuine_n':len(g),'impostor_n':len(i),
                     'false_rejections':int((~accept_scores(g,threshold)).sum()),
                     'false_acceptances':int(accept_scores(i,threshold).sum()),
                     'frr':float((~accept_scores(g,threshold)).mean()),
                     'far':float(accept_scores(i,threshold).mean()),
                     'discrete_eer':curve_metrics(g,i)['eer']},
            'macro':{field:float(np.mean([r[field] for r in rows])) for field in ['far','frr','discrete_eer']},
            'per_identity':rows}


def main():
    # Missing/unfinished report fails before creating artifacts or accessing calibration.
    training=json.loads((RUN/'report.json').read_text())
    training_plan=json.loads((RUN/'plan.json').read_text())
    chosen=completed_checkpoint(training,training_plan)
    protocol_path=ROOT/'research/benchmarks/type2branch_scoring_contract_v1/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    for manifest in [training_plan,protocol]:
        for name,expected in manifest['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError(f'Changed source: {name}')
    checkpoint=Path(chosen['checkpoint']).resolve()
    if checkpoint.parent!=RUN.resolve():raise ValueError('Checkpoint outside frozen run')
    pieces=list(RUN.glob(checkpoint.name+'.*'))
    if not any(p.suffix=='.index' for p in pieces) or not any('.data-' in p.name for p in pieces):
        raise ValueError('Incomplete checkpoint files')
    inner=ROOT/'research/benchmarks/type2branch_inner_train_features_v1'
    roles_path=ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    preparation_plan=json.loads((inner/'plan.json').read_text())
    if sha(roles_path)!=preparation_plan['input_sha256'][str(roles_path.relative_to(ROOT))]:
        raise ValueError('Prepared identity role manifest changed')
    roles=json.loads(roles_path.read_text())
    if set(roles['roles']['calibration'])!=set(preparation_plan['roles']['calibration']):
        raise ValueError('Calibration identities changed')
    array=inner/'calibration.npz'
    expected=json.loads((inner/'report.json').read_text())['roles']['calibration']['sha256']
    if sha(array)!=expected:raise ValueError('Calibration artifact changed')
    OUT.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'scripts/keystroke_benchmark.py',protocol_path,RUN/'report.json',RUN/'plan.json',inner/'report.json',inner/'plan.json',roles_path]
    plan={'checkpoint':chosen,'checkpoint_sha256':{p.name:sha(p) for p in pieces},
          'calibration_sha256':expected,'protocol_sha256':sha(protocol_path),
          'target_far':protocol['target_far'],'scope':'InnerTRAIN calibration only, model choice already frozen',
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__!='1.26.4':raise ValueError('Inference requires pinned NumPy1.26.4')
    ref=ROOT/'research/benchmarks/references/type2branch'
    module('conf',ref/'conf.small.1Kusers.py')
    model=module('type2branch_calibration_model',ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    reader=tf.train.load_checkpoint(str(checkpoint))
    iteration_key='optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
    if not reader.has_tensor(iteration_key) or int(reader.get_tensor(iteration_key))!=chosen['updates']:
        raise ValueError('Checkpoint optimizer update count does not match selected record')
    status=tf.train.Checkpoint(model=model).restore(str(checkpoint))
    status.assert_existing_objects_matched()
    status.expect_partial() # Saved optimizer state is deliberately unused for inference.
    with np.load(array,allow_pickle=False) as a:
        x=a['features'].astype('float32');subjects=a['subject'];windows=a['window']
        assert x.shape==(240,100,5) and (a['true_length']==100).all()
    assert set(subjects)==set(roles['roles']['calibration'])
    embeddings=np.concatenate([model(x[start:start+32],training=False).numpy() for start in range(0,len(x),32)])
    scored=chronological_scores(embeddings,subjects,windows)
    threshold=global_threshold(scored['impostor'],protocol['target_far'])
    result=metrics(scored,threshold)
    assert result['pooled']['genuine_n']==160 and result['pooled']['impostor_n']==2400
    assert result['pooled']['false_acceptances']<=24
    np.savez_compressed(OUT/'calibration_scores.npz',embeddings=embeddings,embedding_subjects=subjects,
                        windows=windows,threshold=np.float64(threshold),**scored)
    report={'status':'inner_train_calibration_complete','threshold':threshold,
            'selected_epoch':chosen['epoch'],'selected_model_is_untrained':chosen['updates']==0,
            'metrics':result,'scores_sha256':sha(OUT/'calibration_scores.npz'),
            'limitations':['Calibration performance is not DEV generalization',
                           'Dependent comparisons; no independent-trial security guarantee',
                           'Three-epoch initial training budget; documented feature/scoring adaptations']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
