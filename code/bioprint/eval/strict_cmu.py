"""Train-only model selection, frozen dev validation, and sealed CMU sessions 7–8.

Never calls eval.cmu.load. The legacy source has been explored historically;
this is explicitly not a pristine test evaluation. All timings after session 6
are skipped before numeric parsing. Usage: python -m eval.strict_cmu --stage train|dev
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import scorer
from eval.cmu import cmu_and_live_names

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).parent / 'data' / 'DSL-StrongPasswordData.csv'
OUT = ROOT / 'research' / 'benchmarks' / 'cmu'
PROTOCOL = {
    'version': 1, 'train_sessions': [1, 2, 3, 4], 'dev_sessions': [5, 6],
    'sealed_test_sessions': [7, 8], 'enrollment': 'first 10 repetitions of session 1',
    'selection': 'sessions 2 and 3; macro per-user EER',
    'calibration': 'session 4; operational thresholds frozen before dev',
    'historical_exposure': 'Legacy CMU scripts previously explored all sessions. No pristine test claim.',
    'population_training': 'Background metric uses sessions 1–2 of OTHER subjects only.',
    'test_measurements_parsed': False,
}


def load_partition(partition: str, source: Path = SOURCE):
    if partition not in {'train', 'dev'}:
        raise ValueError('Test is sealed; only train or dev can be loaded')
    sessions = set(PROTOCOL[f'{partition}_sessions'])
    cmu_names, names = cmu_and_live_names()
    keep = [i for i, name in enumerate(names) if 'Enter#' not in name]
    records = {}
    with source.open(newline='') as stream:
        header = next(csv.reader([next(stream)]))
        if header[3:] != cmu_names:
            raise ValueError('Unexpected CMU columns')
        for line in stream:
            if not line.strip():
                continue
            # Inspect only partition metadata; never parse sealed timing values.
            metadata = line.split(',', 3)
            session = int(metadata[1])
            if session not in sessions:
                continue
            row = next(csv.reader([line]))
            values = [float(row[3 + i]) * 1000 for i in keep]
            records.setdefault(row[0], []).append((session, int(row[2]), values))
    return [names[i] for i in keep], {
        sid: {'session': np.array([r[0] for r in rows]),
              'rep': np.array([r[1] for r in rows]),
              'X': np.array([r[2] for r in rows])}
        for sid, rows in sorted(records.items())
    }


def rates(genuine, impostor, threshold):
    return {'frr': float(np.mean(np.asarray(genuine) > threshold)),
            'far': float(np.mean(np.asarray(impostor) <= threshold))}


def threshold_for(genuine, impostor, target=None):
    g, i = np.sort(genuine), np.sort(impostor)
    thresholds = np.r_[np.nextafter(min(g[0], i[0]), -np.inf), np.unique(np.r_[g, i])]
    frr = 1 - np.searchsorted(g, thresholds, side='right') / len(g)
    far = np.searchsorted(i, thresholds, side='right') / len(i)
    index = (np.argmin(abs(frr - far)) if target is None else
             np.flatnonzero(far <= target + 1e-12)[-1])
    return float(thresholds[index])


def eer(genuine, impostor):
    result = rates(genuine, impostor, threshold_for(genuine, impostor))
    return (result['far'] + result['frr']) / 2


def fit_model(X, names, method, background=None):
    original = scorer.fit(X.tolist(), names)
    model = {'method': method, 'center': original.center, 'spread': original.spread,
             'threshold': original.threshold, 'names': names, 'cap': original.cap}
    if method == 'baseline':
        return model
    if method.startswith('svm-'):
        from sklearn.svm import SVC
        _, c, gamma = method.split('-')
        negative = np.concatenate([rows[:10] for rows in background])
        fit = np.concatenate([X, negative])
        scale = np.maximum(np.std(negative, axis=0), 1.)
        estimator = SVC(C=float(c), gamma=float(gamma) / len(names), class_weight='balanced')
        estimator.fit(fit / scale, np.r_[np.ones(len(X)), np.zeros(len(negative))])
        model.update({'sv': estimator.support_vectors_.tolist(), 'dual': estimator.dual_coef_[0].tolist(),
                      'intercept': float(estimator.intercept_[0]), 'gamma': estimator._gamma,
                      'population_scale': scale.tolist()})
        return model
    if method.startswith('local-'):
        model['reference'] = X.tolist()
        return model
    # Pooled within-user covariance, learned from other TRAINING subjects only.
    residuals = np.concatenate([x - np.median(x, axis=0) for x in background])
    scale = np.maximum(np.std(residuals, axis=0), 1.)
    Z = residuals / scale
    cov = Z.T @ Z / max(len(Z) - 1, 1)
    shrink = float(method.split('-')[-1])
    cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov)) + np.eye(len(names)) * 1e-4
    model['precision'] = np.linalg.inv(cov).tolist()
    model['population_scale'] = scale.tolist()
    model['reference'] = X.tolist()
    return model


def distances(model, X):
    X = np.asarray(X)
    center, spread = np.array(model['center']), np.array(model['spread'])
    method = model['method']
    if method.startswith('svm-'):
        Z = X / np.array(model['population_scale'])
        support = np.array(model['sv'])
        square_distance = np.maximum((Z * Z).sum(axis=1)[:, None] +
                                     (support * support).sum(axis=1)[None, :] - 2 * Z @ support.T, 0)
        return -(np.exp(-model['gamma'] * square_distance) @ np.array(model['dual']) + model['intercept'])
    if method == 'baseline':
        return np.minimum(abs(X - center) / spread, model['cap']).mean(axis=1)
    if method == 'local-l1':
        return (abs(X - center) / spread).mean(axis=1)
    if method == 'local-l2':
        return np.sqrt(np.mean(((X - center) / spread) ** 2, axis=1))
    if method == 'local-knn':
        distances = np.mean(abs(X[:, None, :] - np.array(model['reference'])[None, :, :]) / spread, axis=2)
        return np.sort(distances, axis=1)[:, :3].mean(axis=1)
    precision = np.array(model['precision'])
    scale = np.array(model['population_scale'])
    if method.startswith('pooled-knn'):
        delta = (X[:, None, :] - np.array(model['reference'])[None, :, :]) / scale
        distances = np.sqrt(np.maximum(np.einsum('nrf,fg,nrg->nr', delta, precision, delta), 0))
        return np.sort(distances, axis=1)[:, :3].mean(axis=1)
    delta = (X - center) / scale
    return np.sqrt(np.maximum(np.einsum('nf,fg,ng->n', delta, precision, delta), 0))


def scores_for(models, records, sessions):
    query = {sid: data['X'][np.isin(data['session'], sessions)] for sid, data in records.items()}
    out = {}
    for sid, model in models.items():
        genuine = distances(model, query[sid])
        impostor = distances(model, np.concatenate([x for other, x in query.items() if other != sid]))
        out[sid] = (genuine, impostor)
    return out


def macro(scores):
    return float(np.mean([eer(g, i) for g, i in scores.values()]))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train(enrollment_count=10):
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'protocol.json', PROTOCOL)
    names, records = load_partition('train')
    candidates = ['baseline', 'local-l1', 'local-l2', 'local-knn'] + [
        f'{kind}-{shrink}' for kind in ['pooled-center', 'pooled-knn'] for shrink in [.1, .5, .9]]
    candidates += [f'svm-{c}-{gamma}' for c in [.1, 1, 10] for gamma in [.1, 1, 10]]
    if enrollment_count == 100:
        candidates = ['baseline'] + [f'svm-{c}-{gamma}' for c in [.1, 1, 10] for gamma in [.1, 1, 10]]
    all_models, rows = {}, []
    for method in candidates:
        models = {}
        for sid, data in records.items():
            enrollment = (data['X'][data['session'] <= 2] if enrollment_count == 100 else
                          data['X'][(data['session'] == 1) & (data['rep'] <= 10)])
            background = [other['X'][other['session'] <= 2] for key, other in records.items() if key != sid]
            models[sid] = fit_model(enrollment, names, method, background)
        score = macro(scores_for(models, records, [3] if enrollment_count == 100 else [2, 3]))
        all_models[method] = models
        rows.append({'method': method, 'training_selection_macro_eer': score})
        print(json.dumps(rows[-1]), flush=True)
    selected = min(rows, key=lambda row: row['training_selection_macro_eer'])['method']
    artifact = {'protocol': PROTOCOL, 'candidates': rows, 'selected': selected, 'models': {},
                'elapsed_seconds': time.monotonic() - started}
    for method in dict.fromkeys(['baseline', selected]):
        models = all_models[method]
        calibration = scores_for(models, records, [4])
        for sid, (g, i) in calibration.items():
            models[sid]['calibration'] = {key: threshold_for(g, i, target) for key, target in
                [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
        artifact['models'][method] = models
    write_json(OUT / 'frozen_models.json', artifact)
    print(f'Frozen selection: {selected}; dev and test not loaded', flush=True)


def validate():
    raw = (OUT / 'frozen_models.json').read_bytes()
    artifact = json.loads(raw)
    _, records = load_partition('dev')
    result = {'protocol': PROTOCOL, 'artifact_sha256': hashlib.sha256(raw).hexdigest(),
              'selected_on_training': artifact['selected'], 'split': 'validation_dev', 'results': {}}
    rng = np.random.default_rng(20260920)
    for method, models in artifact['models'].items():
        scores = scores_for(models, records, [5, 6])
        per_user = {}
        for sid, (g, i) in scores.items():
            per_user[sid] = {'eer_diagnostic': eer(g, i), 'genuine_count': len(g), 'impostor_count': len(i),
                            'operating_points': {key: rates(g, i, threshold) for key, threshold in models[sid]['calibration'].items()}}
            if method == 'baseline':
                per_user[sid]['shipped_threshold'] = rates(g, i, models[sid]['threshold'])
        values = np.array([row['eer_diagnostic'] for row in per_user.values()])
        ci = np.quantile(rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1), [.025, .975])
        summary = {'macro_eer_diagnostic': float(values.mean()), 'subject_bootstrap_95pct_ci': ci.tolist(),
                   'operating_points': {key: {rate: float(np.mean([row['operating_points'][key][rate] for row in per_user.values()]))
                        for rate in ['far', 'frr']} for key in ['eer', 'far_1pct', 'far_5pct']}}
        if method == 'baseline':
            summary['shipped_threshold'] = {rate: float(np.mean([row['shipped_threshold'][rate] for row in per_user.values()])) for rate in ['far', 'frr']}
        result['results'][method] = {'summary': summary, 'per_user': per_user}
        print(method, json.dumps(summary), flush=True)
    write_json(OUT / 'dev_results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'dev'], required=True)
    parser.add_argument('--enrollment', type=int, choices=[10, 100], default=10)
    args = parser.parse_args()
    if args.enrollment == 100:
        OUT = OUT.parent / 'cmu100'
        PROTOCOL = {**PROTOCOL, 'enrollment': '100 repetitions across sessions 1 and 2',
                    'selection': 'session 3 only; macro per-user EER',
                    'comparison_warning': 'Higher enrollment budget; compare only to matched 100-sample baseline.'}
    train(args.enrollment) if args.stage == 'train' else validate()
