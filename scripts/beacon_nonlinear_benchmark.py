"""Training-motivated nonlinear paired fusion; previously exposed dev, sealed test."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import joblib
import numpy as np
import beacon_benchmark as source
from device_benchmark import metrics, rates
from eval.strict_cmu import eer

OUT = ROOT / 'research/benchmarks/beacon_nonlinear'
FIT, SELECT, CAL = ['P002', 'P003', 'P006'], ['P007', 'P008'], ['P009', 'P010']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregister():
    OUT.mkdir(exist_ok=True)
    protocol = {
        'version': 1, 'motivation': 'Original frozen training selection EER was weak (approximately41–55%); compare compact nonlinear fusion on unchanged representations.',
        'dev_status': 'Original methods already evaluated on this dev cohort. New methods selected only on training, but this is continued development, not pristine independent validation.',
        'fit_ids': FIT, 'selection_ids': SELECT, 'calibration_ids': CAL,
        'features': 'Exactly original33 paired features; behavior columns0:32, contextadvisory columns0:33. No new transforms.',
        'candidates': [
            {'name': 'original_selected_lr', 'source': 'Original frozen classifier for each modality; no refitting'},
            {'name': 'extra_leaf3', 'class': 'ExtraTreesClassifier', 'n_estimators': 128, 'min_samples_leaf': 3},
            {'name': 'extra_leaf10', 'class': 'ExtraTreesClassifier', 'n_estimators': 128, 'min_samples_leaf': 10},
            {'name': 'rf_depth6_leaf3', 'class': 'RandomForestClassifier', 'n_estimators': 128, 'max_depth': 6, 'min_samples_leaf': 3},
            {'name': 'hgb_leaf7', 'class': 'HistGradientBoostingClassifier', 'max_leaf_nodes': 7, 'l2_regularization': 1, 'max_iter': 100},
            {'name': 'hgb_leaf15', 'class': 'HistGradientBoostingClassifier', 'max_leaf_nodes': 15, 'l2_regularization': 1, 'max_iter': 100}],
        'common': 'random_state20260920; balanced class weights; forests n_jobs1; HGB early_stoppingFalse to avoid hidden internal holdout.',
        'selection': 'Lowest original training-selection EER per behavior/context track; ties prefer candidate order including originalLR first. No refit after selection.',
        'calibration': 'Only originalcalibration identities, target FAR1% and0.1%; freeze before single newdev pass.',
        'test_policy': 'Original metadata split and role ledger; no reserved measurements loaded.',
        'context_policy': 'Advisory comparison only, not physical-person proof or same-device impostor validation.',
        'source_script_sha256': digest(Path(source.__file__)),
        'source_manifest_sha256': digest(source.DATA / 'split_manifest.json'),
        'source_frozen_sha256': digest(source.OUT / 'frozen.json'),
    }
    path = OUT / 'preregistered.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Preregistered protocol changed')
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    return protocol


def check_protocol():
    protocol = json.loads((OUT / 'preregistered.json').read_text())
    assert protocol['source_script_sha256'] == digest(Path(source.__file__))
    assert protocol['source_manifest_sha256'] == digest(source.DATA / 'split_manifest.json')
    assert protocol['source_frozen_sha256'] == digest(source.OUT / 'frozen.json')
    return protocol


def calibration_thresholds(y, scores):
    # Preserve baseline BEACON's observed-score threshold convention, including
    # tie handling; generic tabular next-impostor thresholds differ in score gaps.
    return {str(far): -source.threshold_for(-scores[y == 1], -scores[y == 0], far)
            for far in [.01, .001]}


def repair_calibration():
    """One-time, training-only v1 correction; retain the first frozen artifact."""
    if (OUT / 'frozen_v2.json').exists() or (OUT / 'schema.json').exists():
        raise ValueError('Calibration correction already frozen or dev already exported')
    frozen = json.loads((OUT / 'frozen.json').read_text())
    calibration = np.load(OUT / 'training_calibration_scores.npz', allow_pickle=False)
    for name, spec in frozen['models'].items():
        scores, y = calibration[name], calibration['y']
        spec['thresholds'] = calibration_thresholds(y, scores)
        spec['training_calibration'] = {target: rates(y, scores, threshold)
                                       for target, threshold in spec['thresholds'].items()}
    frozen['version'] = 2
    frozen['correction'] = 'Before dev access: restore original BEACON observed-score threshold convention; no weights or candidate selection changed. Original frozen.json retained.'
    (OUT / 'frozen_v2.json').write_text(json.dumps(frozen, indent=2) + '\n')


def train():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    if (OUT / 'frozen.json').exists():
        raise ValueError('Frozen training exists')
    protocol = check_protocol()
    started = time.monotonic()
    data, excluded = source.load('train')
    assert sorted(data) == sorted(FIT + SELECT + CAL)
    xf, yf, _ = source.pairs({u: data[u] for u in FIT})
    xs, ys, _ = source.pairs({u: data[u] for u in SELECT})
    xc, yc, _ = source.pairs({u: data[u] for u in CAL})
    frozen = {'protocol_sha256': digest(OUT / 'preregistered.json'), 'selection': {}, 'models': {},
              'training_pair_counts': {'fit': len(yf), 'selection': len(ys), 'calibration': len(yc)},
              'training_exclusions': excluded}
    calibration_scores = {}
    for track, width in [('behavior_fusion', 32), ('behavior_plus_context', 33)]:
        models = {'original_selected_lr': joblib.load(source.OUT / f'{track}.joblib')}
        for leaf in [3, 10]:
            models[f'extra_leaf{leaf}'] = ExtraTreesClassifier(n_estimators=128, min_samples_leaf=leaf,
                class_weight='balanced', n_jobs=1, random_state=20260920).fit(xf[:, :width], yf)
        models['rf_depth6_leaf3'] = RandomForestClassifier(n_estimators=128, max_depth=6, min_samples_leaf=3,
            class_weight='balanced', n_jobs=1, random_state=20260920).fit(xf[:, :width], yf)
        for leaf in [7, 15]:
            models[f'hgb_leaf{leaf}'] = HistGradientBoostingClassifier(max_leaf_nodes=leaf, l2_regularization=1,
                max_iter=100, class_weight='balanced', early_stopping=False, random_state=20260920).fit(xf[:, :width], yf)
        selection = {}
        for name, model in models.items():
            distances = -model.predict_proba(xs[:, :width])[:, 1]
            selection[name] = eer(distances[ys == 1], distances[ys == 0])
        chosen = min(models, key=selection.get)
        frozen['selection'][track] = {'candidate_eer': selection, 'selected': chosen}
        # Persist reference too: representation and calibration are matched.
        for label, candidate in [('selected', chosen), ('reference_lr', 'original_selected_lr')]:
            name = f'{track}_{label}'
            selector = ColumnTransformer([('columns', 'passthrough', list(range(width)))]).fit(xf)
            pipeline = make_pipeline(selector, models[candidate])
            scores = pipeline.predict_proba(xc)[:, 1]
            calibration_scores[name] = scores
            thresholds = calibration_thresholds(yc, scores)
            frozen['models'][name] = {'algorithm': candidate, 'width': width, 'artifact': name + '.joblib',
                'thresholds': thresholds, 'training_calibration': {target: rates(yc, scores, threshold)
                                                                for target, threshold in thresholds.items()}}
            joblib.dump(pipeline, OUT / (name + '.joblib'))
    np.savez_compressed(OUT / 'training_calibration_scores.npz', y=yc, **calibration_scores)
    frozen['elapsed_seconds'] = time.monotonic() - started
    (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps(frozen, indent=2))


def evaluate():
    destination = ROOT / 'research/benchmarks/beacon_nonlinear_results.json'
    if destination.exists():
        raise ValueError('New dev pass already recorded')
    protocol = check_protocol()
    frozen = json.loads((OUT / ('frozen_v2.json' if (OUT / 'frozen_v2.json').exists() else 'frozen.json')).read_text())
    assert frozen['protocol_sha256'] == digest(OUT / 'preregistered.json')
    # Existing baseline export already defines the frozen dev pairs exactly.
    with np.load(source.OUT / 'dev_pairs.npz', allow_pickle=False) as data:
        x, y, groups = data['X'], data['y'], data['groups']
    calibration = np.load(OUT / 'training_calibration_scores.npz', allow_pickle=False)
    original_frozen = json.loads((source.OUT / 'frozen.json').read_text())
    original_report = json.loads((source.OUT / 'dev_results.json').read_text())
    result = {'methods': {}, 'test_accessed': False, 'dev_status': protocol['dev_status'],
              'selected_from_training_only': frozen['selection'],
              'limitations': ['Only three dev people; dependent within-session windows.',
                              'Gameplay transfer, not login validation.',
                              'Context may correlate with identity and remains advisory.']}
    schema = {'features': [f'keyboard_difference_{i}' for i in range(16)] +
                         [f'mouse_difference_{i}' for i in range(16)] + ['hardware_context_mismatch'],
              'feature_count': 33, 'score_direction': 'higher_is_genuine', 'models': {},
              'input': 'Exact baseline BEACON33D paired vectors; context advisory only'}
    for name, spec in frozen['models'].items():
        model = joblib.load(OUT / spec['artifact'])
        scores = model.predict_proba(x)[:, 1]
        report = metrics(y, scores, calibration['y'], calibration[name])
        report['operating_points'] = {target: {
            'threshold_selected_on_training_calibration': threshold,
            'train_calibration': rates(calibration['y'], calibration[name], threshold),
            'dev': rates(y, scores, threshold)} for target, threshold in spec['thresholds'].items()}
        if spec['algorithm'] == 'original_selected_lr':
            track = 'behavior_plus_context' if spec['width'] == 33 else 'behavior_fusion'
            original = joblib.load(source.OUT / (track + '.joblib'))
            original_accepts = -original.decision_function(x[:, :spec['width']]) <= original_frozen['models'][track]['thresholds']['far_1pct']
            np.testing.assert_array_equal(scores >= spec['thresholds']['0.01'], original_accepts)
            expected = original_report['results'][track]['operating_points']['far_1pct']
            observed = report['operating_points']['0.01']['dev']
            assert observed['far'] == expected['far'] and observed['frr'] == expected['frr']
            report['original_reference_1pct_decisions_exact'] = True
        result['methods'][name] = report
        schema['models'][name] = {'artifact': spec['artifact'], 'operating_points': report['operating_points']}
    np.savez_compressed(OUT / 'dev_features.npz', X=x, y=y, groups=groups)
    (OUT / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preregister', 'train', 'repair-calibration', 'evaluate'])
    stage = parser.parse_args().stage
    {'preregister': preregister, 'train': train, 'repair-calibration': repair_calibration, 'evaluate': evaluate}[stage]()
