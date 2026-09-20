"""Cross-session verification from real Stroop/Flanker reaction times (Hedge2018).

Three task conditions are not equally spaced numerical difficulty levels. The
condition-index slope is an experimental feature, not an arithmetic slope.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import numpy as np
from device_benchmark import metrics

DATA = ROOT / 'datasets/cognitive'
OUT = ROOT / 'research/benchmarks'
TASKS = ['stroop', 'flanker']


def prepare(download=False):
    DATA.mkdir(parents=True, exist_ok=True)
    metadata_urls = {
        'study1_stroop_metadata.json': 'https://api.osf.io/v2/nodes/cwzds/files/osfstorage/593d96006c613b0229ddd82c/?page%5Bsize%5D=100',
        'study1_flanker_metadata.json': 'https://api.osf.io/v2/nodes/cwzds/files/osfstorage/593d95e6b83f690233314a7c/?page%5Bsize%5D=100',
        'README.upstream.txt': 'https://osf.io/download/e2d34/',
    }
    for name, url in metadata_urls.items():
        path = DATA / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read(2000000))
    files = []
    for task in TASKS:
        metadata = json.loads((DATA / f'study1_{task}_metadata.json').read_text())
        assert metadata['links']['next'] is None, 'Incomplete metadata pagination'
        for item in metadata['data']:
            attrs = item['attributes']
            match = re.fullmatch(r'Study1_P(\d+)(Stroop|Flanker)([12])\.csv', attrs['name'])
            if not match:
                raise ValueError('Unexpected filename, cannot assign metadata split')
            files.append({'subject': match[1], 'task': task, 'session': int(match[3]),
                          'name': attrs['name'], 'bytes': attrs['size'], 'url': item['links']['download'],
                          'sha256': attrs['extra']['hashes']['sha256']})
    subjects = sorted({r['subject'] for r in files}, key=lambda s: hashlib.sha256(('cognitive-v1:' + s).encode()).hexdigest())
    train_end, dev_end = int(.6 * len(subjects)), int(.8 * len(subjects))
    split = {s: 'train' if i < train_end else 'dev' if i < dev_end else 'test' for i, s in enumerate(subjects)}
    for row in files:
        row['split'] = split[row['subject']]
    manifest = {'version': 1, 'source': 'https://osf.io/cwzds/', 'subjects': split, 'files': files,
                'protocol': 'SHA256-ranked study1 participant IDs: firstfloor60% train, nextfloor20% dev, remainder sealed test; session1 enrollment/session2 probe',
                'test_policy': 'metadata only; no test files downloaded', 'planned_bytes': sum(r['bytes'] for r in files if r['split'] != 'test')}
    path = DATA / 'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing changed frozen split')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    roles = {r['name']: {'cohort': r['split'], 'subject': r['subject'],
                         'record_role': 'sealed_test' if r['split'] == 'test' else
                         'training_enrollment_support' if r['session'] == 1 else
                         'dev_validation_probe' if r['split'] == 'dev' else 'training_probe'} for r in files}
    (DATA / 'record_roles.json').write_text(json.dumps(roles, indent=2) + '\n')
    if download:
        def fetch(row):
            assert row['split'] != 'test'
            destination = DATA / row['split'] / row['name']
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raw = destination.read_bytes()
            else:
                for attempt in range(3):
                    try:
                        # Explicit version avoids stale cached signed redirects.
                        url = row['url'] + '?version=1' + ('&direct=1' if attempt else '')
                        with urllib.request.urlopen(url, timeout=60) as response:
                            raw = response.read(row['bytes'] + 1)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if attempt == 2:
                            raise
            if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('Downloaded cognitive file integrity failure: ' + row['name'])
            if not destination.exists():
                destination.write_bytes(raw)
            return len(raw)
        with ThreadPoolExecutor(max_workers=4) as pool:
            sizes = list(pool.map(fetch, [r for r in files if r['split'] != 'test']))
        print(json.dumps({'downloaded_files': len(sizes), 'bytes': sum(sizes)}))
    else:
        print(json.dumps({'subjects': split, 'planned_bytes': manifest['planned_bytes']}))


def load(split):
    if split not in {'train', 'dev'}:
        raise ValueError('Test remains sealed')
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    data = {}
    exclusions = []
    for row in manifest['files']:
        if row['split'] != split:
            continue
        raw = (DATA / split / row['name']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Data changed after split')
        from io import BytesIO
        trials = np.loadtxt(BytesIO(raw), delimiter=',')
        if trials.ndim != 2 or trials.shape[1] != 6:
            raise ValueError('Unknown raw trial schema')
        values = []
        valid = True
        for condition in [0, 1, 2]:
            subset = trials[trials[:, 3] == condition]
            rt = subset[(subset[:, 4] == 1) & (subset[:, 5] >= .1) & (subset[:, 5] <= 5), 5]
            if len(rt) < 10:
                valid = False
                break
            # Seconds from real stimulus presentation; accuracy uses all trials.
            values += [float(np.mean(rt)), float(np.median(rt)), float(np.std(rt)),
                       float(np.quantile(rt, .9)), float(np.mean(subset[:, 4] == 1))]
        if not valid:
            exclusions.append(row['name'])
            continue
        data.setdefault(row['subject'], {}).setdefault(row['session'], {})[row['task']] = np.array(values)
    complete = {s: {session: np.concatenate([v[session][task] for task in TASKS]) for session in [1, 2]}
                for s, v in data.items() if all(session in v and all(t in v[session] for t in TASKS) for session in [1, 2])}
    return complete, {'files_with_insufficient_trials': exclusions, 'incomplete_subjects': sorted(set(data) - set(complete))}


def representation(vector, method):
    means = vector[[0, 5, 10, 15, 20, 25]]
    if method == 'condition_index_slope':
        return (means[[2, 5]] - means[[0, 3]]) / 2
    if method == 'condition_means':
        return means
    return vector


def pairs(data, method):
    x, y = [], []
    for owner in sorted(data):
        for actual in sorted(data):
            x.append(representation(data[actual][2], method) - representation(data[owner][1], method))
            y.append(int(owner == actual))
    return np.array(x), np.array(y)


def evaluate():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    data, exclusions = load('train')
    ordered = sorted(data, key=lambda s: hashlib.sha256(('cognitive-inner:' + s).encode()).hexdigest())
    cut = max(2, int(.7 * len(ordered)))
    fit = {s: data[s] for s in ordered[:cut]}
    calibration = {s: data[s] for s in ordered[cut:]}
    if len(fit) < 5 or len(calibration) < 3:
        raise ValueError('Too few training identities')
    models = {}
    for method in ['condition_index_slope', 'condition_means', 'full_rt_accuracy_profile']:
        xf, yf = pairs(fit, method)
        # Only train fitting identities determine drift and covariance.
        genuine = xf[yf == 1]
        drift = np.median(genuine, axis=0)
        variance = np.maximum(np.var(genuine, axis=0), 1e-6)
        xc, yc = pairs(calibration, method)
        scores = -np.sqrt(np.mean((xc - drift) ** 2 / variance, axis=1))
        models[method] = (drift, variance, scores, yc)
    xf, yf = pairs(fit, 'full_rt_accuracy_profile')
    xc, yc = pairs(calibration, 'full_rt_accuracy_profile')
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                class_weight='balanced', random_state=20260920, n_jobs=2)
    rf.fit(abs(xf), yf)
    rf_cal = rf.predict_proba(abs(xc))[:, 1]
    # All representations/hyperparameters fixed above; dev never tunes anything.
    dev, dev_exclusions = load('dev')
    result = {'version': 1, 'source': 'Hedge et al.2018 Study1 raw Stroop+Flanker', 'test_accessed': False,
              'fit_subjects': sorted(fit), 'calibration_subjects': sorted(calibration), 'dev_subjects': sorted(dev),
              'enrollment_policy': 'Session1 observations are training enrollment support, even for unseen dev-cohort accounts; session2 observations are validation probes. No global fitting on dev cohort.',
              'train_exclusions': exclusions, 'dev_exclusions': dev_exclusions,
              'limitations': ['Standard cognitive-task transfer, not arithmetic or scrambled keypad validation.',
                'Three condition indices do not form an equally spaced difficulty scale; slope is arbitrary contrast.',
                'Uses hundreds of trials per session, not a seconds-long login prompt.',
                'Small participant counts and dependent comparisons limit low-FAR claims.'], 'methods': {}}
    for method, (drift, variance, cal_scores, cal_y) in models.items():
        xd, yd = pairs(dev, method)
        scores = -np.sqrt(np.mean((xd - drift) ** 2 / variance, axis=1))
        result['methods'][method] = metrics(yd, scores, cal_y, cal_scores)
    xd, yd = pairs(dev, 'full_rt_accuracy_profile')
    result['methods']['learned_full_profile_rf'] = metrics(yd, rf.predict_proba(abs(xd))[:, 1], yc, rf_cal)
    artifact_dir = OUT / 'cognitive'
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(rf, artifact_dir / 'learned_full_profile_rf.joblib')
    (artifact_dir / 'schema.json').write_text(json.dumps({
        'features': [f'{task}_{condition}_{stat}' for task in TASKS for condition in ['congruent', 'neutral', 'incongruent']
                     for stat in ['rt_mean', 'rt_median', 'rt_std', 'rt_p90', 'accuracy']],
        'feature_count': 30, 'score_direction': 'higher_is_genuine',
        'input': 'absolute probe-minus-enrollment 30D cognitive feature difference',
        'models': {'learned_full_profile_rf': {'artifact': 'learned_full_profile_rf.joblib',
                      'operating_points': result['methods']['learned_full_profile_rf']['operating_points']}},
        'distance_models': {name: {'drift': drift.tolist(), 'variance': variance.tolist(),
                                  'operating_points': result['methods'][name]['operating_points']}
                            for name, (drift, variance, _, _) in models.items()},
    }, indent=2) + '\n')
    np.savez_compressed(artifact_dir / 'dev_features.npz', X=abs(xd), signed_difference=xd, y=yd)
    (OUT / 'cognitive_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'download', 'evaluate'])
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate()
    else:
        prepare(args.command == 'download')
