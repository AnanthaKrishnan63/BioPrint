"""Balabit session-separated train/calibration/dev benchmark. Test reads prohibited.

Run with bigidea Python and PYTHONPATH=.research-deps:code/bioprint.
All model choice and operating thresholds use training calibration sessions only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/bioprint'))
from engine.scorer import _fit_arrays, _deviations

DATA = ROOT / 'data/benchmarks/balabit'
OUT = ROOT / 'research/benchmarks/pointer'

def features(block):
    t, x, y = block.T
    dt = np.diff(t)
    dx, dy = np.diff(x), np.diff(y)
    good = (dt > 0) & (dt < 1)
    if good.sum() < 32:
        return None
    dt, dx, dy = dt[good], dx[good], dy[good]
    length = np.hypot(dx, dy)
    speed = length / dt
    vx, vy = dx / dt, dy / dt
    angle = np.arctan2(dy, dx)
    turn = np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))
    accel = np.diff(speed) / dt[1:]
    values = []
    for sequence in (speed, np.abs(vx), np.abs(vy), accel, np.abs(turn)):
        values.extend([float(np.mean(sequence)), float(np.std(sequence)), *np.quantile(sequence, [.1, .5, .9]).tolist()])
    path = length.sum()
    values.extend([float(np.hypot(dx.sum(), dy.sum()) / max(path, 1e-9)), float((turn > 0).mean()), float(np.abs(turn).sum() / max(path, 1)), float(np.max(speed) / max(speed.mean(), 1e-9))])
    return values

def load_entries(entries):
    X, labels, groups = [], [], []
    for entry in entries:
        if entry['split'] == 'test_sealed':
            raise ValueError('Refusing test content')
        path = DATA / entry['path']
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if blob != entry['sha']:
            raise ValueError(f'Source integrity failure: {path}')
        rows = csv.DictReader(content.decode().splitlines())
        points = []
        for row in rows:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row['state'] != 'Move':
                continue
            t, x, y = float(row['client timestamp']), float(row['x']), float(row['y'])
            if points and (t <= points[-1][0] or t - points[-1][0] > 5):
                points = []
            points.append((t, x, y))
            if len(points) == 128:
                vector = features(np.asarray(points))
                if vector is not None:
                    X.append(vector)
                    labels.append(entry['path'].split('/')[1])
                    groups.append(entry['path'])
                points = []
    return np.asarray(X), np.asarray(labels), np.asarray(groups)

def stats(y, s, threshold):
    positive, negative = s[y == 1], s[y == 0]
    fpr, tpr, _ = roc_curve(y, s)
    index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return {'far': float((negative >= threshold).mean()), 'frr': float((positive < threshold).mean()), 'eer_descriptive': float((fpr[index] + 1 - tpr[index]) / 2), 'auc': float(roc_auc_score(y, s)), 'genuine_comparisons': len(positive), 'impostor_comparisons': len(negative)}

def threshold_at_far(y, s, target):
    negatives = s[y == 0]
    descending = np.sort(negatives)[::-1]
    allowed = int(np.floor(target * len(descending)))
    return float(np.nextafter(descending[min(allowed, len(descending) - 1)], np.inf))

def scores(name, train, labels, query, users, save_models=False):
    result = []
    names = [f'pointer.{i}' for i in range(train.shape[1])]
    for user in users:
        if name == 'existing_scaled_manhattan':
            center, spread = _fit_arrays(train[labels == user], np.asarray(names))
            s = -_deviations(center, spread, query, 6.0).mean(axis=1)
            if save_models:
                np.savez_compressed(OUT / f"{name}_{user}.npz", center=center, spread=spread, cap=6.0)
        else:
            cls = RandomForestClassifier if name.startswith('rf') else ExtraTreesClassifier
            leaf = int(name.split('_')[-1])
            model = cls(n_estimators=160, min_samples_leaf=leaf, max_features='sqrt', class_weight='balanced', n_jobs=2, random_state=20260920)
            model.fit(train, labels == user)
            s = model.predict_proba(query)[:, 1]
            if save_models:
                joblib.dump(model, OUT / f"{name}_{user}.joblib", compress=3)
        result.append(s)
    return np.asarray(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Explicit rerun only; never tune from dev results.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists() and not args.force:
        raise SystemExit('Results already exist; inspect logs rather than repeating.')
    start = time.time()
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    entries = manifest['entries']
    train, labels, train_groups = load_entries([e for e in entries if e['split'] == 'train' and e['train_role'] == 'fit'])
    cal, cal_labels, cal_groups = load_entries([e for e in entries if e['split'] == 'train' and e['train_role'] == 'calibration'])
    users = sorted(set(labels))
    # Bound CPU use and user imbalance using evenly spaced training-only rows.
    indices = np.concatenate([np.flatnonzero(labels == u)[np.linspace(0, (labels == u).sum() - 1, min(1000, (labels == u).sum())).astype(int)] for u in users])
    train, labels = train[indices], labels[indices]
    truth = np.asarray([cal_labels == u for u in users]).ravel().astype(int)
    variants = ['existing_scaled_manhattan', 'rf_1', 'rf_5', 'extra_1', 'extra_5']
    calibration = {}
    thresholds = {}
    for name in variants:
        s = scores(name, train, labels, cal, users).ravel()
        threshold = threshold_at_far(truth, s, .01)
        thresholds[name] = {str(far): threshold_at_far(truth, s, far) for far in (.001, .01, .05)}
        calibration[name] = stats(truth, s, threshold)
        print(name, calibration[name], flush=True)
    selected = min(variants[1:], key=lambda n: (calibration[n]['frr'], calibration[n]['eer_descriptive']))
    # Freeze choices on disk before any dev measurement is accessed.
    frozen = {'selected': selected, 'thresholds': thresholds, 'calibration': calibration, 'variants': variants}
    (OUT / 'frozen_training_selection.json').write_text(json.dumps(frozen, indent=2))
    dev, dev_labels, dev_groups = load_entries([e for e in entries if e['split'] == 'dev'])
    dev_truth = np.asarray([dev_labels == u for u in users]).ravel().astype(int)
    reports = {}
    for name in ['existing_scaled_manhattan', selected]:
        s_matrix = scores(name, train, labels, dev, users, save_models=True)
        s = s_matrix.ravel()
        reports[name] = {}
        for far, threshold in thresholds[name].items():
            pooled = stats(dev_truth, s, threshold)
            per_user = {u: stats((dev_labels == u).astype(int), s_matrix[i], threshold) for i, u in enumerate(users)}
            pooled['per_claimed_user'] = per_user
            pooled['macro_far'] = float(np.mean([v['far'] for v in per_user.values()]))
            pooled['macro_frr'] = float(np.mean([v['frr'] for v in per_user.values()]))
            pooled['macro_eer_descriptive'] = float(np.mean([v['eer_descriptive'] for v in per_user.values()]))
            reports[name][far] = pooled
        np.savez_compressed(OUT / f'{name}_dev_scores.npz', scores=s_matrix, true_user=dev_labels, users=np.asarray(users), sessions=dev_groups)
    result = {'dataset': 'Balabit official training subset only', 'window': '128 consecutive Move events; no session overlap', 'feature_count': train.shape[1], 'split': manifest['protocol'], 'selected_from_training_only': selected, 'calibration': calibration, 'dev': reports, 'counts': {'fit_windows': len(train), 'calibration_windows': len(cal), 'dev_windows': len(dev), 'fit_sessions': len(set(train_groups)), 'calibration_sessions': len(set(cal_groups)), 'dev_sessions': len(set(dev_groups)), 'sealed_sessions_not_downloaded': sum(e['split'] == 'test_sealed' for e in entries)}, 'elapsed_seconds': time.time() - start, 'limitations': ['Ten users; correlated within-session windows; no population-level confidence claim.', 'Remote-desktop data; device/task confounds unresolved.', 'Existing scorer evaluated on compatible trajectory descriptors, not production submit-target extractor.', 'EER is dev descriptive; operating thresholds frozen from train calibration only.', 'No official test files or reserved test measurements accessed.']}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
