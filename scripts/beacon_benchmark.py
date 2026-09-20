"""Aligned keyboard/mouse fusion on a bounded BEACON subset, test never read.

This is a gameplay-to-login transfer experiment, not login accuracy. Train-only
subject groups fit/select/calibrate; unseen dev identities enroll on first session.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from eval.strict_cmu import threshold_for, rates, eer
DATA = ROOT / 'datasets/beacon'
OUT = ROOT / 'research/benchmarks/beacon'
WINDOW = 30.0
MIN_KEYS = 5
MIN_POINTS = 32


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.] * 5
    return [float(x) for x in np.r_[np.quantile(values, [.1, .5, .9]), np.mean(values), np.std(values)]]


def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))


def read_csv(path, partial=False):
    raw = path.read_bytes()
    if partial:
        raw = raw[:raw.rfind(b'\n') + 1]
    return list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))


def session_features(files):
    if any(row['split'] not in {'train', 'dev'} for row in files):
        raise ValueError('Test is sealed')
    by = {row['modality']: row for row in files}
    keyboard = by['keyboard_csv']
    mouse = by['mouse_csv']
    krows = read_csv(DATA / keyboard['split'] / keyboard['release_path'])
    mrows = read_csv(DATA / mouse['split'] / mouse['release_path'], mouse['partial'])
    # Public raw timestamps mix epoch and relative clocks in TRAINING data.
    # Use recorder-elapsed columns consistently; never infer alignment from labels.
    keys = np.array([[float(r['Elapsed Start Time']), float(r['Elapsed Release Time'])] for r in krows], float).reshape(-1, 2)
    points = np.array([[float(r['Elapsed Time']), float(r['X']), float(r['Y'])] for r in mrows if r['Event'] == 'Move'], float).reshape(-1, 3)
    if not len(keys) or not len(points):
        return [], {}
    keys = keys[np.argsort(keys[:, 0], kind='stable')]
    points = points[np.argsort(points[:, 0], kind='stable')]
    # Coalesce repeated timestamps without creating zero-dt velocities.
    keep = np.r_[np.diff(points[:, 0]) > 0, True]
    points = points[keep]
    lo, hi = max(keys[0, 0], points[0, 0]), min(keys[-1, 1], points[-1, 0])
    output = []
    for start in np.arange(lo, hi - WINDOW + 1e-8, WINDOW):
        k = keys[(keys[:, 0] >= start) & (keys[:, 1] < start + WINDOW)]
        p = points[(points[:, 0] >= start) & (points[:, 0] < start + WINDOW)]
        if len(k) < MIN_KEYS or len(p) < MIN_POINTS:
            continue
        duration = k[:, 1] - k[:, 0]
        dd = np.diff(k[:, 0])
        ud = k[1:, 0] - k[:-1, 1]
        key_vector = summary(duration) + summary(dd) + summary(ud) + [float(np.mean(ud < 0))]
        dt = np.diff(p[:, 0]); delta = np.diff(p[:, 1:], axis=0)
        valid = (dt > 0) & (dt < 1)
        dt, delta = dt[valid], delta[valid]
        if len(dt) < MIN_POINTS:
            continue
        speed = np.linalg.norm(delta, axis=1) / dt
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        turn = np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))
        acceleration = np.diff(speed) / dt[1:]
        mouse_vector = summary(speed) + summary(acceleration) + summary(abs(turn)) + [
            float(np.linalg.norm(delta.sum(axis=0)) / max(np.linalg.norm(delta, axis=1).sum(), 1.))]
        output.append({'start': float(start), 'keyboard': signed_log(np.array(key_vector)).tolist(),
                       'mouse': signed_log(np.array(mouse_vector)).tolist()})
    hardware = json.loads((DATA / by['hardware']['split'] / by['hardware']['release_path']).read_text()) if 'hardware' in by else {}
    # Coarse context only, exclude serial IDs/device names; never label this identity evidence.
    hardware = {k: hardware.get(k) for k in ['OS', 'OS_Version', 'Architecture', 'Processor', 'RAM', 'Monitors']}
    return output, hardware


def load(split):
    if split not in {'train', 'dev'}:
        raise ValueError('Test is sealed')
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    grouped = {}
    for row in manifest['files']:
        if row['split'] != split:
            continue
        grouped.setdefault((row['participant_id'], row['role']), []).append(row)
    data = {}
    for (subject, role), files in sorted(grouped.items()):
        windows, hardware = session_features(files)
        data.setdefault(subject, {})[role] = {'windows': windows, 'hardware': hardware}
    # Deterministic quality requirements declared before dev; exclusions are reported.
    valid = {s: d for s, d in data.items() if len(d.get('enrollment', {}).get('windows', [])) >= 2 and d.get('probe', {}).get('windows')}
    excluded = {s: {role: len(value['windows']) for role, value in d.items()} for s, d in data.items() if s not in valid}
    return valid, excluded


def pairs(data):
    X, y, groups = [], [], []
    profiles = {}
    for owner, recordings in data.items():
        profiles[owner] = {}
        for modality in ['keyboard', 'mouse']:
            vectors = np.array([w[modality] for w in recordings['enrollment']['windows']])
            center = np.median(vectors, axis=0)
            spread = np.maximum(np.mean(abs(vectors - center), axis=0), .15)
            profiles[owner][modality] = (center, spread)
    for owner, profile in profiles.items():
        for actual, recordings in data.items():
            hardware_a = data[owner]['enrollment']['hardware']
            hardware_b = recordings['probe']['hardware']
            context = [float(hardware_a[k] != hardware_b[k]) for k in hardware_a if k in hardware_b]
            context = float(np.mean(context)) if context else 0.
            for window in recordings['probe']['windows']:
                row = []
                for modality in ['keyboard', 'mouse']:
                    center, spread = profile[modality]
                    row += np.minimum(abs(np.array(window[modality]) - center) / spread, 8).tolist()
                X.append(row + [context])
                y.append(int(owner == actual))
                groups.append((owner, actual))
    return np.array(X), np.array(y), groups


FEATURES = {'keyboard': list(range(16)), 'mouse': list(range(16, 32)),
            'behavior_fusion': list(range(32)), 'behavior_plus_context': list(range(33))}


def measure(y, scores, thresholds):
    g, i = scores[y == 1], scores[y == 0]
    return {'eer_diagnostic': eer(g, i), 'genuine_comparisons': len(g), 'impostor_comparisons': len(i),
            'operating_points': {k: rates(g, i, t) for k, t in thresholds.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'dev'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'train':
        if (OUT / 'frozen.json').exists():
            raise SystemExit('Frozen artifact exists; inspect logs instead of rerunning')
        started = time.monotonic()
        data, excluded = load('train')
        ids = sorted(data)
        if len(ids) < 6:
            raise ValueError(f'Need >=6 training identities for fit/select/calibrate; got {ids}, excluded={excluded}')
        fit_ids, select_ids, cal_ids = ids[:-4], ids[-4:-2], ids[-2:]
        xf, yf, _ = pairs({u: data[u] for u in fit_ids})
        xs, ys, _ = pairs({u: data[u] for u in select_ids})
        xc, yc, _ = pairs({u: data[u] for u in cal_ids})
        frozen = {'manifest_sha256': hashlib.sha256((DATA / 'split_manifest.json').read_bytes()).hexdigest(),
                  'protocol': {'window_seconds': WINDOW, 'minimum_keys': MIN_KEYS, 'minimum_mouse_points': MIN_POINTS,
                               'fit_identities': fit_ids, 'selection_identities': select_ids, 'calibration_identities': cal_ids,
                               'dev': 'unseen identities; first session enrollment, second session probes', 'sealed_test_read': False},
                  'train_exclusions': excluded, 'models': {}, 'candidates': {}}
        for modality, cols in FEATURES.items():
            candidates = []
            for c in [.01, .1, 1.]:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=20260920))
                model.fit(xf[:, cols], yf)
                score = -model.decision_function(xs[:, cols])
                value = eer(score[ys == 1], score[ys == 0])
                candidates.append((value, c, model))
            value, c, model = min(candidates, key=lambda item: item[0])
            # Do not refit after selection; thresholds calibrated for this exact model.
            scores = -model.decision_function(xc[:, cols])
            thresholds = {key: threshold_for(scores[yc == 1], scores[yc == 0], target) for key, target in [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
            frozen['models'][modality] = {'columns': cols, 'C': c, 'thresholds': thresholds, 'selection_eer': value}
            frozen['candidates'][modality] = [{'C': cc, 'selection_eer': ee} for ee, cc, _ in candidates]
            joblib.dump(model, OUT / f'{modality}.joblib')
        baseline = xc[:, :32].mean(axis=1)
        frozen['baseline_thresholds'] = {key: threshold_for(baseline[yc == 1], baseline[yc == 0], target) for key, target in [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
        frozen['elapsed_seconds'] = time.monotonic() - started
        (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
        print(json.dumps(frozen, indent=2))
    else:
        frozen = json.loads((OUT / 'frozen.json').read_text())
        if hashlib.sha256((DATA / 'split_manifest.json').read_bytes()).hexdigest() != frozen['manifest_sha256']:
            raise ValueError('Dataset protocol changed after fitting')
        data, excluded = load('dev')
        if len(data) < 2:
            raise ValueError(f'Insufficient dev identities; excluded={excluded}')
        x, y, groups = pairs(data)
        results = {'subjects': sorted(data), 'exclusions': excluded, 'results': {}, 'protocol': frozen['protocol'],
                   'limitations': ['Gameplay, not login; only three planned dev identities.', 'Context may correlate with identity through own-device use; no same-device impostor validation.', 'Only a bounded prefix of each paired session used.', 'Keypad and malicious bots are not measured in this dataset.', 'Dev enrollment builds personal templates, not global weights or thresholds.']}
        for modality, config in frozen['models'].items():
            model = joblib.load(OUT / f'{modality}.joblib')
            score = -model.decision_function(x[:, config['columns']])
            results['results'][modality] = measure(y, score, config['thresholds'])
        baseline = x[:, :32].mean(axis=1)
        results['results']['untrained_equal_behavior_fusion'] = measure(y, baseline, frozen['baseline_thresholds'])
        np.savez_compressed(OUT / 'dev_pairs.npz', X=x, y=y, groups=np.asarray(groups))
        (OUT / 'dev_results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
