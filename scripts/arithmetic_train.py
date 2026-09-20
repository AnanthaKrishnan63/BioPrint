"""Frozen TRAIN-only arithmetic representation/metric comparison."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import sklearn
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve
from arithmetic_features import profile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/arithmetic'
OUT = ROOT / 'research/benchmarks/arithmetic_train_v1'
FEATURES = ['mean_bounded_wait', 'ordinal_wait_slope', 'condition_bounded_wait',
            'wait_missing_error_profile']
METHODS = [(feature, metric) for feature in FEATURES for metric in ['scaled_l1', 'shrinkage_mahalanobis']]
TARGETS = [.001, .01, .05]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def records(role, acquisition):
    if role not in ('train_fit', 'train_calibration'):
        raise ValueError('TRAIN loader forbids DEV/test')
    rows = [r for r in acquisition['entries'] if r['role'] == role]
    if len(rows) != (7 if role == 'train_fit' else 4):
        raise ValueError('Unexpected TRAIN cohort')
    if any(Path(r['path']).name != r['path'] for r in rows):
        raise ValueError('Archive path escape')
    return sorted(rows, key=lambda r: r['path'])


def load_train(role, acquisition, receipt):
    rows = records(role, acquisition)
    hashes = {r['path']: r['sha256'] for r in receipt['archives']}
    features = {key: [] for key in FEATURES}
    accounts = []
    for row in rows:
        path = DATA / 'raw' / row['path']
        if sha(path) != hashes[row['path']]:
            raise ValueError('Archive changed')
        account = path.stem
        rounds = []
        with zipfile.ZipFile(path) as archive:
            for trial in range(2, 7):
                blocks = {}
                for level in 'lmh':
                    member = f'Cal_{account}_L{level}T{trial}.mat'
                    if archive.getinfo(member).file_size > 2_000_000:
                        raise ValueError('Expanded member exceeds bound')
                    blocks[level] = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)['Data']
                rounds.append(profile(blocks))
        for key in FEATURES:
            features[key].append([r[key] for r in rounds])
        accounts.append(account)
    return accounts, {key: np.asarray(value, dtype=float) for key, value in features.items()}


def fit_metric(support):
    if support.ndim != 3 or support.shape[1] != 2 or not np.isfinite(support).all():
        raise ValueError('Expected finite account x two support rounds x features')
    spread = support.reshape(-1, support.shape[-1]).std(axis=0)
    active = spread > 0
    if not active.any():
        raise ValueError('All dimensions constant')
    scaled = support[:, :, active] / spread[active]
    residuals = (scaled - scaled.mean(axis=1, keepdims=True)).reshape(-1, active.sum())
    estimator = LedoitWolf(assume_centered=True).fit(residuals)
    if not np.isfinite(estimator.precision_).all() or np.linalg.eigvalsh(estimator.covariance_).min() <= 0:
        raise ValueError('Degenerate within-account covariance')
    return {'active': active.tolist(), 'spread': spread[active].tolist(),
            'precision': estimator.precision_.tolist(), 'shrinkage': float(estimator.shrinkage_)}


def score(metric, parameters, templates, queries):
    active = np.asarray(parameters['active'], dtype=bool)
    spread = np.asarray(parameters['spread'])
    if templates.ndim != 2 or queries.ndim != 2 or templates.shape[1] != len(active) or queries.shape[1] != len(active):
        raise ValueError('Invalid input feature dimensions')
    if not np.isfinite(templates).all() or not np.isfinite(queries).all():
        raise ValueError('Nonfinite features')
    delta = (templates[:, None, active] - queries[None, :, active]) / spread
    if metric == 'scaled_l1':
        return -np.abs(delta).mean(axis=-1)
    if metric != 'shrinkage_mahalanobis':
        raise ValueError('Unknown metric')
    squared = np.einsum('aqd,de,aqe->aq', delta, np.asarray(parameters['precision']), delta)
    return -np.sqrt(np.maximum(squared, 0))


def labels(accounts, rounds):
    actual = np.repeat(accounts, rounds)
    return (np.asarray(accounts)[:, None] == actual[None, :]).astype(int).ravel()


def threshold(y, scores, target):
    negative = np.sort(scores[y == 0])[::-1]
    if not len(negative):
        raise ValueError('No calibration impostors')
    index = int(np.floor(target * len(negative)))
    return float(np.nextafter(negative[index], np.inf))


def stats(y, values, cutoff):
    positive, negative = values[y == 1], values[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError('Both claim classes required')
    fpr, tpr, _ = roc_curve(y, values)
    idx = np.argmin(np.abs(fpr - (1 - tpr)))
    return dict(far=float(np.mean(negative >= cutoff)), frr=float(np.mean(positive < cutoff)),
                eer=float((fpr[idx] + 1 - tpr[idx]) / 2), genuine=len(positive), impostor=len(negative))


def choose(results):
    names = list(results)
    def rank(name):
        r = results[name]
        return (r['far'] > .01, r['far'] if r['far'] > .01 else 0., r['frr'], r['eer'], names.index(name))
    selected = min(names, key=rank)
    r = results[selected]
    status = 'no_low_far_qualifier' if r['far'] > .01 else 'no_useful_discrimination' if r['frr'] == 1 else 'selected'
    return selected, status


def prepare():
    acquisition = json.loads((DATA / 'plan.json').read_text())
    for role in ('train_fit', 'train_calibration'):
        records(role, acquisition)
    OUT.mkdir(exist_ok=False)
    sources = ['arithmetic_train.py', 'arithmetic_features.py']
    plan = dict(inputs={name: sha(DATA / name) for name in ['plan.json', 'complete.json']},
                sources={name: sha(ROOT / 'scripts' / name) for name in sources},
                sklearn_version=sklearn.__version__, methods=METHODS, targets=TARGETS,
                parameter_fit='Seven TRAIN-fit identities, T2/T3 only: global std and LedoitWolf within-account residual covariance',
                internal_threshold='TRAIN-fit T4 at1% target; no T5/T6 thresholds',
                selection='TRAIN-fit T5/T6; prefer FAR<=1%, then FRR, EER, fixed order. If none qualifies, minimum FAR then FRR/EER; mark failure explicitly.',
                final_calibration='Four disjoint TRAIN-calibration identities; T2/T3 personal support, T4-T6 threshold calibration; no global parameter refit or method reselection',
                observation_budget='30 enrollment trials (two blocks per level),15 probe trials (one block per level); levels from noncontemporaneous ordered blocks',
                role_records={role: records(role, acquisition) for role in ['train_fit', 'train_calibration']},
                failure_policy='Fail whole cohort for any missing/unparseable block; do not silently drop records',
                interpretation='Source-backed distance/covariance adaptation, not established cognitive-authentication SOTA; bounded wait is not uncensored RT',
                dev_accessed=False, test_accessed=False)
    write(OUT / 'plan.json', plan)
    for name in sources:
        (OUT / ('source_' + name)).write_bytes((ROOT / 'scripts' / name).read_bytes())


def run():
    if (OUT / 'selection.json').exists():
        raise FileExistsError('TRAIN execution already started; preserve history')
    plan = json.loads((OUT / 'plan.json').read_text())
    for name, expected in plan['inputs'].items():
        if sha(DATA / name) != expected: raise ValueError('Input changed')
    for name, expected in plan['sources'].items():
        if sha(ROOT / 'scripts' / name) != expected: raise ValueError('Source changed')
    acquisition = json.loads((DATA / 'plan.json').read_text())
    receipt = json.loads((DATA / 'complete.json').read_text())
    for role, expected in plan['role_records'].items():
        if records(role, acquisition) != expected: raise ValueError('Role records changed')
    accounts, features = load_train('train_fit', acquisition, receipt)
    parameters, results, arrays = {}, {}, {}
    for feature in FEATURES:
        parameters[feature] = fit_metric(features[feature][:, :2])
    for feature, metric in METHODS:
        name = feature + '__' + metric
        x = features[feature]; gallery = x[:, :2].mean(axis=1)
        calibration = score(metric, parameters[feature], gallery, x[:, 2]).ravel()
        selection = score(metric, parameters[feature], gallery, x[:, 3:].reshape(-1, x.shape[-1])).ravel()
        cutoff = threshold(labels(accounts, 1), calibration, .01)
        results[name] = stats(labels(accounts, 2), selection, cutoff)
        results[name]['internal_threshold'] = cutoff
        arrays[name + '_internal_calibration'] = calibration
        arrays[name + '_selection'] = selection
    selected, status = choose(results)
    write(OUT / 'selection.json', dict(selected=selected, status=status, methods=results,
                                      parameters=parameters, plan_sha256=sha(OUT / 'plan.json')))
    # This boundary makes selection immutable before new calibration identities.
    cal_accounts, cal_features = load_train('train_calibration', acquisition, receipt)
    thresholds, calibration_results = {}, {}
    for feature, metric in METHODS:
        name = feature + '__' + metric
        x = cal_features[feature]; gallery = x[:, :2].mean(axis=1)
        scores = score(metric, parameters[feature], gallery, x[:, 2:].reshape(-1, x.shape[-1])).ravel()
        y = labels(cal_accounts, 3)
        thresholds[name] = {str(target): threshold(y, scores, target) for target in TARGETS}
        calibration_results[name] = {key: stats(y, scores, cutoff) for key, cutoff in thresholds[name].items()}
        arrays[name + '_final_calibration'] = scores
    np.savez_compressed(OUT / 'scores.npz', fit_accounts=np.asarray(accounts),
                        calibration_accounts=np.asarray(cal_accounts), **arrays)
    write(OUT / 'train_complete.json', dict(plan=plan, selected=selected, selection_status=status,
        selection=results, parameters=parameters, thresholds=thresholds, calibration=calibration_results,
        scores_sha256=sha(OUT / 'scores.npz'), selection_sha256=sha(OUT / 'selection.json'),
        dev_accessed=False, test_accessed=False))
    print(json.dumps({'selected': selected, 'selection_status': status, 'selection': results,
                      'calibration': calibration_results[selected]}))


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
