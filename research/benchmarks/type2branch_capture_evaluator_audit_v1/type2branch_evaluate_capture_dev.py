"""Score the frozen short-capture DEV protocol; no model/threshold selection."""
import json
import os
from pathlib import Path
import sys
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.research-type2branch-deps'))
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_calibrate_lengths import verify_hashes
from type2branch_capture_calibration_gate import calibrated_capture
from type2branch_capture_metrics import capture_metrics
from type2branch_reference_smoke import module
from type2branch_scoring import gallery_scores

OUT = ROOT/'research/benchmarks/type2branch_capture_dev_evaluation_v1'


def assemble_gallery(embeddings, subjects, windows, identities):
    embeddings = np.asarray(embeddings); subjects = np.asarray(subjects); windows = np.asarray(windows)
    if (embeddings.ndim != 2 or subjects.shape != (len(embeddings),) or windows.shape != subjects.shape
            or set(subjects) != set(identities) or not np.isfinite(embeddings).all()):
        raise ValueError('Complete finite gallery cohort required')
    grouped = []
    for subject in identities:
        rows = np.flatnonzero(subjects == subject); rows = rows[np.argsort(windows[rows])]
        if not np.array_equal(windows[rows], np.arange(15,20)):
            raise ValueError('Exactly enrollment windows15..19 required')
        grouped.append(embeddings[rows])
    return np.stack(grouped)


def main():
    features = ROOT/'research/benchmarks/type2branch_capture_dev_features_v1'
    prepared = json.loads((features/'report.json').read_text())
    if prepared['status'] != 'frozen_capture_dev_features_complete' or prepared['failures']:
        raise ValueError('Complete prescribed DEV preparation required; no subset scoring')
    preparation_plan = json.loads((features/'plan.json').read_text())
    verify_hashes(preparation_plan)
    protocol = preparation_plan['protocol']; verify_hashes(protocol)
    calibration = ROOT/'research/benchmarks/type2branch_length_calibration_v1'
    calibrated = json.loads((calibration/'report.json').read_text())
    cal_plan = json.loads((calibration/'plan.json').read_text()); verify_hashes(cal_plan)
    verify_hashes(json.loads((ROOT/'research/benchmarks/type2branch_train_v1/plan.json').read_text()))
    pair = ROOT/'research/benchmarks/type2branch_length_pair_v1'
    comparison = json.loads((pair/'report.json').read_text())
    reports = {arm: json.loads((ROOT/f'research/benchmarks/type2branch_length_train_{arm}_v1/report.json').read_text())
               for arm in ['full','mixed']}
    thresholds = calibrated_capture(comparison, reports, calibrated, cal_plan)
    if ({int(k):v for k,v in preparation_plan['thresholds'].items()} != thresholds
            or preparation_plan['checkpoint'] != comparison['selected_checkpoint']):
        raise ValueError('Prepared checkpoint/threshold binding changed')
    checkpoint = Path(cal_plan['checkpoint']['checkpoint'])
    for name, expected in cal_plan['checkpoint_sha256'].items():
        if sha(checkpoint.parent/name) != expected: raise ValueError('Checkpoint changed')
    for role in ['gallery','probes']:
        if sha(features/f'{role}.npz') != prepared['roles'][role]['sha256']:
            raise ValueError('Prepared DEV artifact changed')
    identities = np.array(sorted(protocol['allowed_identities']))
    if len(identities) != 79 or set(prepared['coverage']) != set(identities):
        raise ValueError('All79identities must remain in coverage')
    sources = [Path(__file__), features/'report.json', features/'plan.json', calibration/'report.json',
               calibration/'plan.json', pair/'report.json']
    sources += [ROOT/'scripts'/name for name in ['type2branch_capture_metrics.py',
        'type2branch_capture_calibration_gate.py', 'type2branch_scoring.py',
        'type2branch_reference_smoke.py','keystroke_benchmark.py']]
    plan = {'protocol': protocol, 'checkpoint': comparison['selected_checkpoint'], 'thresholds': thresholds,
        'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sources},
        'input_sha256': {str((features/f'{r}.npz').relative_to(ROOT)):prepared['roles'][r]['sha256'] for r in ['gallery','probes']}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='', TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2', TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__ != '1.26.4': raise ValueError('Pinned NumPy1.26.4 required')
    ref = ROOT/'research/benchmarks/references/type2branch'
    module('conf',ref/'conf.small.1Kusers.py')
    model = module('capture_dev_model',ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    reader=tf.train.load_checkpoint(str(checkpoint));key='optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
    if not reader.has_tensor(key) or int(reader.get_tensor(key)) != plan['checkpoint']['updates']:
        raise ValueError('Saved optimizer iteration mismatch')
    restored=tf.train.Checkpoint(model=model).restore(str(checkpoint))
    restored.assert_existing_objects_matched();restored.expect_partial()
    arrays={}
    for role in ['gallery','probes']:
        with np.load(features/f'{role}.npz',allow_pickle=False) as data:
            x=data['features'].astype('float32');subjects=data['subject'];windows=data['window'];lengths=data['true_length']
        if x.shape != (len(subjects),100,5) or not np.isfinite(x).all():
            raise ValueError('Invalid model inputs')
        if role=='gallery' and not (lengths==100).all(): raise ValueError('Full gallery windows required')
        if role=='probes' and not (windows==0).all(): raise ValueError('First capture only')
        embedded=np.concatenate([model(x[start:start+32],training=False).numpy() for start in range(0,len(x),32)]) if len(x) else np.empty((0,256))
        arrays[role]=(embedded,subjects,windows,lengths)
    ge,gs,gw,_=arrays['gallery'];pe,ps,pw,lengths=arrays['probes']
    gallery=assemble_gallery(ge,gs,gw,identities)
    scores=gallery_scores(pe,gallery) if len(pe) else np.empty((0,len(identities)))
    result=capture_metrics(scores,ps,lengths,identities,prepared['coverage'],thresholds,protocol['cohorts'])
    row_thresholds=np.array([thresholds[int(length)] for length in lengths],dtype=np.float64)
    decisions=scores>=row_thresholds[:,None]
    target=OUT/'scores.npz'
    np.savez_compressed(target,scores=scores,decisions=decisions,subjects=identities,
        probe_subjects=ps,probe_lengths=lengths,thresholds=row_thresholds,
        gallery_embeddings=gallery,probe_embeddings=pe)
    report={'status':'frozen_capture_dev_evaluation_complete','metrics':result,
        'scores_sha256':sha(target),'selected_checkpoint':plan['checkpoint'],
        'sealed_test_payloads_decoded':False,'limitations':protocol['limitations'],
        'prior_exposure':protocol['prior_exposure']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
