"""FPStalker browser-linkage research benchmark; never opens reserved test values.

Run with bigidea Python. `prepare` reads identity metadata first and only decodes
feature fields for training/dev identities. `evaluate` has no test-set option.
This is an independent FPStalker-inspired implementation, not a paper replication.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
sys.path.insert(0, str(ROOT / 'code/bioprint'))
DATA = ROOT / 'datasets/fpstalker'
RESULTS = ROOT / 'research/benchmarks'
SEED = 'bioprint-device-v1'
PREFIX = re.compile(rb"^\((\d+),\s*'([^']+)',")
TOKEN = re.compile(r"\s*('(?:\\.|[^'\\])*'|NULL|-?\d+(?:\.\d+)?)(?:,|\))", re.S)
COLUMNS = ['counter', 'id', 'addressHttp', 'creationDate', 'updateDate', 'endDate',
           'userAgentHttp', 'acceptHttp', 'hostHttp', 'connectionHttp', 'encodingHttp',
           'languageHttp', 'orderHttp', 'pluginsJS', 'platformJS', 'cookiesJS', 'dntJS',
           'timezoneJS', 'resolutionJS', 'localJS', 'sessionJS', 'IEDataJS', 'canvasJS',
           'webGLJs', 'fontsFlash', 'resolutionFlash', 'languageFlash', 'platformFlash',
           'adBlock', 'vendorWebGLJS', 'rendererWebGLJS', 'octaneScore', 'sunspiderTime',
           'pluginsJSHashed', 'canvasJSHashed', 'webGLJsHashed', 'fontsFlashHashed',
           'osDetailed', 'browserDetailed', 'browserVersion']
FEATURES = ['userAgentHttp', 'languageHttp', 'platformJS', 'cookiesJS', 'dntJS',
            'timezoneJS', 'resolutionJS', 'localJS', 'sessionJS', 'vendorWebGLJS',
            'rendererWebGLJS', 'pluginsJSHashed', 'canvasJSHashed', 'webGLJsHashed',
            'osDetailed', 'browserDetailed', 'browserVersion']
FUZZY = ['userAgentHttp', 'languageHttp', 'rendererWebGLJS', 'browserVersion']


def bucket(identity: str, salt: str = SEED) -> int:
    return int(hashlib.sha256(f'{salt}:{identity}'.encode()).hexdigest()[:8], 16) % 100


def partition(identity: str) -> str:
    n = bucket(identity)
    return 'train' if n < 60 else 'dev' if n < 80 else 'test'


def rows():
    """Stream SQL text as opaque lines; no SQL execution or disk extraction."""
    for name in ['extension1.txt.tar.gz', 'extension2.txt.tar.gz']:
        with tarfile.open(DATA / name) as archive:
            for member in archive:
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        for line in stream:
                            if line.startswith(b'('):
                                yield line


def prepare():
    counts = Counter()
    for raw in rows():
        match = PREFIX.match(raw)
        if not match:
            raise ValueError('Unexpected row prefix; refusing to inspect feature values')
        counts[match[2].decode()] += 1
    manifest = {
        'version': 1, 'seed': SEED,
        'protocol': 'SHA256 identity-disjoint train/dev/test 60/20/20 before feature decoding',
        'test_policy': 'Reserved in upstream archives; metadata counts only; never decoded/evaluated',
        'subjects': {k: {'split': partition(k), 'rows': v} for k, v in sorted(counts.items())},
        'archives': {p.name: {'bytes': p.stat().st_size,
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted(DATA.glob('*.tar.gz'))},
    }
    target = DATA / 'split_manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Existing split differs; refusing to overwrite')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    outputs = {s: (DATA / f'{s}.jsonl').open('w') for s in ['train', 'dev']}
    parsed = Counter()
    try:
        for raw in rows():
            identity = PREFIX.match(raw)[2].decode()
            split = partition(identity)
            if split == 'test':
                continue  # do not decode or tokenize ANY test feature values
            text = raw.decode('utf-8')
            values = []
            cursor = 1
            while len(values) < len(COLUMNS):
                m = TOKEN.match(text, cursor)
                if not m:
                    raise ValueError(f'Malformed permitted SQL row at field {len(values)}')
                value = m[1]
                if value.startswith("'"):
                    value = re.sub(r'\\(.)', lambda x: {'n': '\n', 'r': '\r', 't': '\t', '0': '\0'}.get(x[1], x[1]), value[1:-1])
                elif value == 'NULL':
                    value = None
                values.append(value)
                cursor = m.end()
            if text[cursor:].strip() not in {',', ';'}:
                raise ValueError('Unexpected row suffix or schema mismatch')
            full = dict(zip(COLUMNS, values))
            keep = {k: full[k] for k in ['counter', 'id', 'creationDate'] + FEATURES}
            outputs[split].write(json.dumps(keep, separators=(',', ':')) + '\n')
            parsed[split] += 1
    finally:
        for output in outputs.values():
            output.close()
    write_record_roles()
    print(json.dumps({'prepared_rows': parsed, 'reserved_test_rows': sum(v for k, v in counts.items() if partition(k) == 'test'),
                      'subjects': dict(Counter(partition(k) for k in counts))}))


def load(split):
    if split not in {'train', 'dev'}:
        raise ValueError('Test values must remain sealed')
    groups = defaultdict(list)
    with (DATA / f'{split}.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            assert partition(row['id']) == split
            groups[row['id']].append(row)
    # Retain subjects with >=3 observations; first is enrollment, later probes.
    return {key: sorted(values, key=lambda x: (x['creationDate'], int(x['counter'])))
            for key, values in groups.items() if len(values) >= 3}


def write_record_roles():
    roles = {}
    for cohort in ['train', 'dev']:
        for subject, records in load(cohort).items():
            for i, row in enumerate(records):
                roles[row['counter']] = {'cohort': cohort,
                    'record_role': 'training_enrollment_support' if i == 0 else
                    'dev_validation_probe' if cohort == 'dev' else 'training_probe'}
    (DATA / 'record_roles.json').write_text(json.dumps({
        'policy': 'First chronological observation is training enrollment support, independent of global-fit identity cohort; no reserved test feature access',
        'records': roles}, indent=2) + '\n')


def pair_features(a, b):
    values = [float(a[k] == b[k]) for k in FEATURES]
    values += [SequenceMatcher(None, a[k] or '', b[k] or '', autojunk=False).ratio() for k in FUZZY]
    values += [sum(values[:len(FEATURES)]) / len(FEATURES)]
    return values


def env(row):
    """Only direct collector overlap. No IP, fontsFlash, invented hardware fields."""
    value = {'probe_version': 'fpstalker-adapter-v1', 'ua': row['userAgentHttp'], 'platform': row['platformJS'],
             'languages': (row['languageHttp'] or '').split(','),
             'canvas_hash': row['canvasJSHashed'], 'webgl_vendor': row['vendorWebGLJS'],
             'webgl_renderer': row['rendererWebGLJS'],
             'cookies_enabled': str(row['cookiesJS']).lower() in {'yes', 'true', '1'},
             'do_not_track': row['dntJS']}
    dims = re.findall(r'\d+', row['resolutionJS'] or '')
    if len(dims) >= 2:
        value['screen_width'], value['screen_height'] = map(int, dims[:2])
    if len(dims) >= 3:
        value['color_depth'] = int(dims[2])
    return value


def make_pairs(groups, impostors=20):
    """At most 20 later probes/browser, each against its enrollment and 20 others."""
    import numpy as np
    identities = sorted(groups)
    rng = np.random.default_rng(20260920)
    pairs, labels, owners = [], [], []
    for identity in identities:
        eligible = [x for x in identities if x != identity]
        for probe in groups[identity][1:21]:
            pairs.append((groups[identity][0], probe))
            labels.append(1)
            owners.append(identity)
            for other in rng.choice(eligible, min(impostors, len(eligible)), replace=False):
                pairs.append((groups[str(other)][0], probe))
                labels.append(0)
                owners.append(identity)
    return pairs, np.array(labels), owners


def threshold_at_far(labels, scores, far):
    import numpy as np
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed = int(np.floor(far * len(negatives)))
    # Strictly exclude the next unallowed negative, correctly handling ties.
    return float(np.nextafter(negatives[allowed], np.inf)) if allowed < len(negatives) else float('-inf')


def rates(labels, scores, threshold):
    import numpy as np
    neg, pos = labels == 0, labels == 1
    fa = int(np.sum(scores[neg] >= threshold))
    fr = int(np.sum(scores[pos] < threshold))
    return {'far': fa / int(neg.sum()), 'frr': fr / int(pos.sum()),
            'false_accepts': fa, 'false_rejects': fr,
            'impostor_trials': int(neg.sum()), 'genuine_trials': int(pos.sum())}


def metrics(labels, scores, calibration_labels, calibration_scores):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Interpolated descriptive dev EER only; never selects deployed threshold.
    crossing = np.flatnonzero(fpr >= fnr)
    i = int(crossing[0])
    if i == 0:
        eer = (fpr[0] + fnr[0]) / 2
    else:
        d0, d1 = fpr[i-1] - fnr[i-1], fpr[i] - fnr[i]
        weight = -d0 / (d1 - d0)
        eer = fpr[i-1] + weight * (fpr[i] - fpr[i-1])
    result = {'dev_eer_descriptive': float(eer), 'operating_points': {}}
    for far in [.01, .001]:
        threshold = threshold_at_far(calibration_labels, calibration_scores, far)
        result['operating_points'][str(far)] = {
            'threshold_selected_on_training_calibration': threshold,
            'train_calibration': rates(calibration_labels, calibration_scores, threshold),
            'dev': rates(labels, scores, threshold),
        }
    return result


def evaluate():
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from engine import device
    train = load('train')
    fit = {k: v for k, v in train.items() if bucket(k, SEED + ':inner') < 75}
    calibration = {k: v for k, v in train.items() if k not in fit}
    dev = load('dev')
    sets = {}
    for name, groups in [('fit', fit), ('calibration', calibration), ('dev', dev)]:
        pairs, y, owners = make_pairs(groups)
        x = np.array([pair_features(a, b) for a, b in pairs])
        live_results = [device.check(env(b), [env(a)]) for a, b in pairs]
        assert all(r.available for r in live_results), 'Unavailable signals cannot count as measurements'
        live = np.array([-r.score for r in live_results])
        sets[name] = {'x': x, 'y': y, 'live': live, 'owners': owners}
    output = {'dataset': 'FPStalker public 15k sample', 'version': 2,
              'task': 'unseen-browser identity linkage, NOT person/physical-device verification',
              'test_accessed': False,
              'selection': 'prespecified models; no dev-based tuning or threshold selection',
              'eligible_subjects': {k: len(v) for k, v in [('fit', fit), ('calibration', calibration), ('dev', dev)]},
              'feature_names': FEATURES + [k + '_similarity' for k in FUZZY] + ['mean_equal'],
              'methods': {}}
    score_sets = {}
    # ThresholdFP (2025), Algorithm 2 weighting, adapted to pair verification.
    # Compute all within-browser pair changes on fit identities only. Algebraic
    # counts avoid quadratic memory; no lineage/remediation claim is made.
    changes = np.zeros(len(FEATURES), dtype=float)
    total_changed = 0
    for values in fit.values():
        n = len(values)
        full_counts = Counter(tuple(v[k] for k in FEATURES) for v in values)
        total_changed += n * n - sum(c * c for c in full_counts.values())
        for j, key in enumerate(FEATURES):
            counts = Counter(v[key] for v in values)
            changes[j] += n * n - sum(c * c for c in counts.values())
    stability = 1 - changes / max(total_changed, 1)
    output['thresholdfp_adapted_stability_weights'] = dict(zip(FEATURES, stability.tolist()))
    for name in sets:
        x = sets[name]['x']
        score_sets[name] = {'equal_attribute_agreement': x[:, :len(FEATURES)].mean(axis=1),
                            'exact_fingerprint_match': np.all(x[:, :len(FEATURES)] == 1, axis=1).astype(float),
                            'bioprint_overlap_device_score': sets[name]['live'],
                            'thresholdfp_2025_pairwise_adaptation': -((1 - x[:, :len(FEATURES)]) @ stability)}
    models = {
        'fpstalker_inspired_random_forest': RandomForestClassifier(n_estimators=160, min_samples_leaf=5,
                            max_depth=12, class_weight='balanced', n_jobs=2, random_state=20260920),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                            l2_regularization=1.0, random_state=20260920),
    }
    for name, model in models.items():
        model.fit(sets['fit']['x'], sets['fit']['y'])
        for split in ['calibration', 'dev']:
            score_sets[split][name] = model.predict_proba(sets[split]['x'])[:, 1]
    for name, scores in score_sets['dev'].items():
        output['methods'][name] = metrics(sets['dev']['y'], scores, sets['calibration']['y'], score_sets['calibration'][name])
    output['methods']['bioprint_overlap_device_score']['original_threshold_dev'] = rates(sets['dev']['y'], sets['dev']['live'], -device.THRESHOLD_BITS)
    RESULTS.mkdir(exist_ok=True)
    artifact_dir = RESULTS / 'device'
    artifact_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, artifact_dir / f'{name}.joblib')
    (artifact_dir / 'schema.json').write_text(json.dumps({
        'features': output['feature_names'], 'feature_count': len(output['feature_names']),
        'score_direction': 'higher_is_genuine', 'input': 'pair similarity vector',
        'thresholdfp_weights': stability.tolist(),
        'models': {name: {'artifact': name + '.joblib', 'operating_points': output['methods'][name]['operating_points']} for name in models},
    }, indent=2) + '\n')
    np.savez_compressed(artifact_dir / 'dev_features.npz', X=sets['dev']['x'], y=sets['dev']['y'], owners=np.array(sets['dev']['owners']))
    (RESULTS / 'device_results.json').write_text(json.dumps(output, indent=2) + '\n')
    np.savez_compressed(RESULTS / 'device_dev_scores.npz', labels=sets['dev']['y'], owners=np.array(sets['dev']['owners']), **score_sets['dev'])
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate}[args.command]()
