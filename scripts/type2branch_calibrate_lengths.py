"""Calibrate four raw-prefix lengths only after both TRAIN arms complete."""
import json
import os
from pathlib import Path
import sys
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.research-type2branch-deps'))
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_length_completion import completed_pair
from type2branch_calibrate import metrics
from type2branch_scoring import chronological_scores, global_threshold
from type2branch_reference_smoke import module

OUT = ROOT/'research/benchmarks/type2branch_length_calibration_v1'


def verify_hashes(manifest):
    for group in ['source_sha256', 'input_sha256']:
        for name, expected in manifest.get(group, {}).items():
            if sha(ROOT/name) != expected:
                raise ValueError(f'Frozen dependency changed: {name}')


def main():
    pair = ROOT/'research/benchmarks/type2branch_length_pair_v1'
    # Require all reports before reading calibration arrays or creating output.
    comparison = json.loads((pair/'report.json').read_text())
    runs = {arm: ROOT/f'research/benchmarks/type2branch_length_train_{arm}_v1'
            for arm in ['full', 'mixed']}
    reports = {arm: json.loads((run/'report.json').read_text()) for arm, run in runs.items()}
    chosen = completed_pair(comparison, reports)
    receipts = {}
    for arm, run in runs.items():
        receipt_path = ROOT/f'research/benchmarks/type2branch_length_{arm}_receipt_v1/report.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['status'] != 'completed_arm_scores_verified_and_files_receipted'
                or receipt['arm'] != arm or receipt['report_sha256'] != sha(run/'report.json')
                or receipt['plan_sha256'] != sha(run/'plan.json')):
            raise ValueError('Completed-arm receipt mismatch')
        receipts[arm] = receipt
    verify_hashes(json.loads((pair/'plan.json').read_text()))
    for run in runs.values():
        verify_hashes(json.loads((run/'plan.json').read_text()))
    run = runs[comparison['selected_arm']]
    checkpoint = Path(chosen['checkpoint']).resolve()
    if checkpoint.parent != run.resolve():
        raise ValueError('Selected checkpoint outside completed arm')
    pieces = sorted(run.glob(checkpoint.name+'.*'))
    if {p.name for p in pieces} != {checkpoint.name+'.index', checkpoint.name+'.data-00000-of-00001'}:
        raise ValueError('Expected complete single-shard checkpoint')
    receipted = receipts[comparison['selected_arm']]['checkpoints'][str(chosen['epoch'])]
    if (receipted['checkpoint'] != str(checkpoint) or receipted['updates'] != chosen['updates']
            or receipted['files'] != {p.name: sha(p) for p in pieces}):
        raise ValueError('Selected checkpoint differs from completed-arm receipt')
    features = ROOT/'research/benchmarks/type2branch_length_calibration_features_v1'
    feature_plan = json.loads((features/'plan.json').read_text())
    feature_report = json.loads((features/'report.json').read_text())
    if (feature_plan['lengths'] != [25, 50, 75, 100]
            or feature_report['status'] != 'raw_prefix_calibration_features_prepared'):
        raise ValueError('Completed four-length feature preparation required')
    verify_hashes(feature_plan)
    for length in feature_plan['lengths']:
        if sha(features/f'length_{length}.npz') != feature_report['lengths'][str(length)]['sha256']:
            raise ValueError('Calibration features changed')
    # The author model/config/backend remain those pinned by original training.
    original_plan = ROOT/'research/benchmarks/type2branch_train_v1/plan.json'
    verify_hashes(json.loads(original_plan.read_text()))
    sources = [Path(__file__), pair/'report.json', pair/'plan.json', original_plan,
               features/'plan.json', features/'report.json']
    sources += [run/name for run in runs.values() for name in ['plan.json', 'report.json']]
    sources += [ROOT/f'research/benchmarks/type2branch_length_{arm}_receipt_v1/report.json' for arm in runs]
    sources += [ROOT/'scripts'/name for name in ['type2branch_length_completion.py',
        'type2branch_length_training_core.py', 'type2branch_calibrate.py',
        'type2branch_scoring.py', 'keystroke_benchmark.py', 'type2branch_reference_smoke.py']]
    plan = {'selected_arm': comparison['selected_arm'], 'checkpoint': chosen,
        'lengths': feature_plan['lengths'], 'target_far': .01,
        'scope': 'Reserved TRAIN calibration only; one inclusive threshold per exact probe length',
        'checkpoint_sha256': {p.name: sha(p) for p in pieces},
        'input_sha256': {str((features/f'length_{length}.npz').relative_to(ROOT)):
                        feature_report['lengths'][str(length)]['sha256'] for length in feature_plan['lengths']},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='', TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2', TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    # Imported benchmark utilities prepend general dependencies; restore the
    # pinned TensorFlow overlay before loading TensorFlow and its dependencies.
    sys.path.insert(0, str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__ != '1.26.4': raise ValueError('Pinned NumPy1.26.4 required')
    ref = ROOT/'research/benchmarks/references/type2branch'
    module('conf', ref/'conf.small.1Kusers.py')
    model = module('length_calibration_model', ref/'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH': 100, 'INPUT_FEATURES': 5})['model']
    reader = tf.train.load_checkpoint(str(checkpoint))
    key = 'optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
    if not reader.has_tensor(key) or int(reader.get_tensor(key)) != chosen['updates']:
        raise ValueError('Checkpoint optimizer iteration mismatch')
    restored = tf.train.Checkpoint(model=model).restore(str(checkpoint))
    restored.assert_existing_objects_matched(); restored.expect_partial()
    results = {}; reference_gallery = None
    for length in plan['lengths']:
        with np.load(features/f'length_{length}.npz', allow_pickle=False) as data:
            x = data['features'].astype('float32'); subjects = data['subject']; windows = data['window']
            gallery = windows < 5
            if (x.shape != (240, 100, 5) or set(subjects) != set(feature_plan['identities'])
                    or not (data['true_length'][gallery] == 100).all()
                    or not (data['true_length'][~gallery] == length).all()):
                raise ValueError('Calibration array contract changed')
        if reference_gallery is None: reference_gallery = x[gallery].copy()
        else: np.testing.assert_array_equal(reference_gallery, x[gallery])
        embeddings = np.concatenate([model(x[start:start+32], training=False).numpy()
                                     for start in range(0, len(x), 32)])
        scored = chronological_scores(embeddings, subjects, windows)
        threshold = global_threshold(scored['impostor'], plan['target_far'])
        result = metrics(scored, threshold)
        if (result['pooled']['genuine_n'] != 160 or result['pooled']['impostor_n'] != 2400
                or result['pooled']['false_acceptances'] > 24):
            raise ValueError('Calibration denominator or target-FAR mismatch')
        target = OUT/f'length_{length}_scores.npz'
        np.savez_compressed(target, embeddings=embeddings, embedding_subjects=subjects,
                            windows=windows, threshold=np.float64(threshold), **scored)
        results[str(length)] = {'threshold': threshold, 'metrics': result, 'scores_sha256': sha(target)}
    report = {'status': 'paired_length_train_calibration_complete', 'selected_arm': plan['selected_arm'],
        'selected_checkpoint': chosen, 'lengths': results, 'dev_test_observations_decoded': False,
        'limitations': ['Calibration is same-session TRAIN, not cross-session DEV performance',
                       'Length variants and all-claim scores are dependent comparisons',
                       'Raw-prefix calibration differs from feature-prefix augmentation',
                       'Empirical 1% calibration FAR is not a future security guarantee']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
