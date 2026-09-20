"""Preregistered touch landing/timing verification; no sealed row decoding."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import numpy as np
from device_benchmark import metrics, rates, threshold_at_far

DATA = ROOT / 'datasets/touch_tsi'
OUT = ROOT / 'research/benchmarks/touch_tsi'
PREFIX = re.compile(rb'^(user\d{2}),(task[1-4]),(\d+),')
NAMES = ['landing_dx_mean', 'landing_dx_std', 'landing_dx_median',
         'landing_dy_mean', 'landing_dy_std', 'landing_dy_median',
         'intertap_ms_mean', 'intertap_ms_std', 'intertap_ms_median', 'intertap_ms_p90']


def download():
    # Existing metadata/protocol are versioned with the benchmark. Never acquire
    # an unbounded archive or silently accept a changed publisher object.
    prereg = json.loads((OUT / 'preregistered.json').read_text())
    tree = json.loads((DATA / 'source_tree.json').read_text())
    assert tree['sha'] == prereg['source_tree']
    entries = [entry for entry in tree['tree'] if entry['path'] in {'touch_data.csv', 'keyboard_data.json'}]
    assert sum(entry['size'] for entry in entries) < 500000000
    for entry in entries:
        destination = DATA / entry['path']
        if destination.exists():
            continue
        url = ('https://raw.githubusercontent.com/google-research-datasets/'
               'tap-typing-with-touch-sensing-images/main/' + entry['path'] + '?download=1')
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(entry['size'] + 1)
        digest = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if len(raw) != entry['size'] or digest != entry['sha']:
            raise ValueError('Publisher object changed; download rejected')
        destination.write_bytes(raw)
    verify_and_manifest()


def role(subject, task, trial):
    if not 1 <= int(subject[4:]) <= 12 or task == 'task4':
        return 'sealed_test'
    if task == 'task1':
        return 'training_enrollment_support' if trial < 15 else 'training_fit_probe'
    if task == 'task2':
        return 'training_selection' if trial < 15 else 'training_calibration'
    return 'dev_probe'


def verify_and_manifest():
    prereg = json.loads((OUT / 'preregistered.json').read_text())
    tree = json.loads((DATA / 'source_tree.json').read_text())
    assert tree['sha'] == prereg['source_tree']
    receipts = {}
    for name in ['touch_data.csv', 'keyboard_data.json']:
        entry = next(item for item in tree['tree'] if item['path'] == name)
        path = DATA / name
        size = path.stat().st_size
        digest, sha256 = hashlib.sha1(), hashlib.sha256()
        digest.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                digest.update(chunk)
                sha256.update(chunk)
        assert size == entry['size'] and digest.hexdigest() == entry['sha'], name
        receipts[name] = {'bytes': size, 'git_blob_sha1': digest.hexdigest(), 'sha256': sha256.hexdigest()}
    counts = defaultdict(int)
    trial_roles = {}
    with (DATA / 'touch_data.csv').open('rb') as stream:
        next(stream)
        for line in stream:
            match = PREFIX.match(line)
            if not match:
                raise ValueError('Unknown row prefix; refusing measurement decoding')
            subject, task, trial = match.groups()
            assigned = role(subject.decode(), task.decode(), int(trial))
            counts[assigned] += 1
            trial_roles[f'{subject.decode()}/{task.decode()}/{int(trial)}'] = assigned
    manifest = {'protocol': prereg, 'files': receipts, 'metadata_row_counts': dict(counts),
                'test_access': 'Only row participant/task/trial prefix inspected; no sealed measurement decoding'}
    path = DATA / 'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Frozen manifest changed')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    (DATA / 'record_roles.json').write_text(json.dumps(trial_roles, indent=2) + '\n')
    print(json.dumps({'files': receipts, 'row_roles': counts}))


def load(roles):
    allowed = {'training_enrollment_support', 'training_fit_probe', 'training_selection', 'training_calibration', 'dev_probe'}
    if not set(roles) <= allowed:
        raise ValueError('Sealed roles may not be loaded')
    keys = json.loads((DATA / 'keyboard_data.json').read_text())['keys_info']
    rows = defaultdict(list)
    with (DATA / 'touch_data.csv').open('rb') as stream:
        header = next(csv.reader([next(stream).decode()]))
        for raw in stream:
            match = PREFIX.match(raw)
            if not match:
                raise ValueError('Unknown row prefix')
            subject, task, trial = match.groups()
            subject, task, trial = subject.decode(), task.decode(), int(trial)
            assigned = role(subject, task, trial)
            if assigned not in roles:
                continue  # no decoding of dev in training, nor any test measurements
            row = dict(zip(header, next(csv.reader([raw.decode()]))))
            key = keys.get(row['ref_char'])
            if key is None:
                raise ValueError('No documented key geometry: ' + row['ref_char'])
            values = [float(row['timestamp_ms']),
                (float(row['first_frame_touch_x']) - key['key_center_x']) / key['key_width'],
                (float(row['first_frame_touch_y']) - key['key_center_y']) / key['key_height']]
            if np.isfinite(values).all():
                rows[(subject, task, trial, assigned)].append(values)
    result, excluded = [], []
    for identity, points in sorted(rows.items()):
        values = trial_features(points)
        if values is None:
            excluded.append(identity)
        else:
            result.append({'subject': identity[0], 'task': identity[1], 'trial': identity[2],
                           'role': identity[3], 'x': values})
    return result, excluded


def trial_features(points):
    x = np.asarray(sorted(points), dtype=float)
    if len(x) < 5:
        return None
    gaps = np.diff(x[:, 0])
    gaps = gaps[(gaps > 0) & (gaps <= 5000)]
    if len(gaps) < 3:
        return None
    return np.array([fn(x[:, axis]) for axis in [1, 2] for fn in [np.mean, np.std, np.median]] +
                    [np.mean(gaps), np.std(gaps), np.median(gaps), np.quantile(gaps, .9)])


def pairs(records, templates):
    x, y, groups = [], [], []
    for row in records:
        for owner, template in sorted(templates.items()):
            x.append(abs(row['x'] - template))
            y.append(int(row['subject'] == owner))
            groups.append(row['subject'])
    return np.array(x), np.array(y), np.array(groups)


def train():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if (OUT / 'frozen.json').exists():
        raise ValueError('Frozen training exists; refusing retraining')
    train_roles = ['training_enrollment_support', 'training_fit_probe', 'training_selection', 'training_calibration']
    records, excluded = load(train_roles)
    templates = {user: np.mean([r['x'] for r in records if r['subject'] == user and
                              r['role'] == 'training_enrollment_support'], axis=0)
                 for user in sorted({r['subject'] for r in records})}
    assert len(templates) == 12 and all(np.isfinite(v).all() for v in templates.values())
    split = {name: pairs([r for r in records if r['role'] == name], templates) for name in train_roles[1:]}
    xf, yf, _ = split['training_fit_probe']
    xs, ys, _ = split['training_selection']
    xc, yc, _ = split['training_calibration']
    models = {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=20260920)),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=20260920, n_jobs=2),
    }
    selection = {}
    for name, model in models.items():
        model.fit(xf, yf)
        selection[name] = metrics(ys, model.predict_proba(xs)[:, 1], yf, model.predict_proba(xf)[:, 1])['dev_eer_descriptive']
    chosen = min(models, key=selection.get)
    model = models[chosen]
    cal = model.predict_proba(xc)[:, 1]
    fitting = np.array([r['x'] for r in records if r['role'] == 'training_fit_probe'])
    scale = np.maximum(fitting[:, :6].std(axis=0), 1e-6)
    baseline_cal = -(xc[:, :6] / scale).mean(axis=1)
    frozen = {'selected': chosen, 'selection_eer': selection, 'features': NAMES,
              'templates': {k: v.tolist() for k, v in templates.items()}, 'baseline_scale': scale.tolist(),
              'thresholds': {name: {str(far): threshold_at_far(yc, scores, far) for far in [.01, .001]}
                             for name, scores in [('selected', cal), ('landing_baseline', baseline_cal)]},
              'train_calibration': {name: {str(far): rates(yc, scores, threshold_at_far(yc, scores, far)) for far in [.01, .001]}
                                    for name, scores in [('selected', cal), ('landing_baseline', baseline_cal)]},
              'train_exclusions': excluded, 'training_trial_counts': {name: sum(r['role'] == name for r in records) for name in train_roles},
              'protocol_sha256': hashlib.sha256((OUT / 'preregistered.json').read_bytes()).hexdigest()}
    joblib.dump(model, OUT / 'selected.joblib')
    np.savez_compressed(OUT / 'calibration.npz', X=xc, y=yc)
    (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps({'selected': chosen, 'selection_eer': selection, 'counts': frozen['training_trial_counts'], 'excluded': len(excluded)}))


def evaluate():
    import joblib
    if (ROOT / 'research/benchmarks/touch_tsi_results.json').exists():
        raise ValueError('Dev already evaluated')
    frozen = json.loads((OUT / 'frozen.json').read_text())
    assert frozen['protocol_sha256'] == hashlib.sha256((OUT / 'preregistered.json').read_bytes()).hexdigest()
    records, excluded = load(['dev_probe'])
    templates = {k: np.array(v) for k, v in frozen['templates'].items()}
    x, y, groups = pairs(records, templates)
    model = joblib.load(OUT / 'selected.joblib')
    calibration = np.load(OUT / 'calibration.npz', allow_pickle=False)
    scale = np.array(frozen['baseline_scale'])
    scores = {'selected': model.predict_proba(x)[:, 1], 'landing_baseline': -(x[:, :6] / scale).mean(axis=1)}
    cal_scores = {'selected': model.predict_proba(calibration['X'])[:, 1], 'landing_baseline': -(calibration['X'][:, :6] / scale).mean(axis=1)}
    methods = {name: metrics(y, score, calibration['y'], cal_scores[name]) for name, score in scores.items()}
    for name, report in methods.items():
        for target, values in report['operating_points'].items():
            assert values['threshold_selected_on_training_calibration'] == frozen['thresholds'][name][target]
    report = {'source': 'Google TSI UIST2024', 'selected_model': frozen['selected'], 'methods': methods,
              'dev_trials': len(records), 'dev_participants': len(set(groups)), 'dev_exclusions': excluded,
              'test_accessed': False, 'limitations': json.loads((OUT / 'preregistered.json').read_text())['nonclaims']}
    np.savez_compressed(OUT / 'dev_features.npz', X=x, y=y, groups=groups)
    schema = {'feature_count': 10, 'features': NAMES, 'input': 'Absolute probe-trial minus enrollment-template differences',
              'score_direction': 'higher_is_genuine', 'models': {'selected': {'artifact': 'selected.joblib',
              'operating_points': methods['selected']['operating_points']}}, 'selected_algorithm': frozen['selected']}
    (OUT / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    (ROOT / 'research/benchmarks/touch_tsi_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'prepare', 'train', 'evaluate'])
    command = parser.parse_args().command
    {'download': download, 'prepare': verify_and_manifest, 'train': train, 'evaluate': evaluate}[command]()
