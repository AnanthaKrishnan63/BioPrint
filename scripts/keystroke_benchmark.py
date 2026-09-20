"""Train-only model selection, then frozen KeyRecs session-2 validation.

Example: bigidea/bin/python scripts/keystroke_benchmark.py --track fixed
Test identities are inaccessible through the data adapter. Outputs are DEV
results, not untouched test estimates or a reproduction of published SOTA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np

from data_keystrokes import DATA, checksum, iter_rows, prepare


def load_track(track: str, split: str, window: int = 50):
    """Keep chronological acquisition order; never overlap free-text windows.

    Nonfinite values become missing and are imputed from fitting data only.
    Free-text five-channel summaries omit key identities and text entirely.
    No window crosses a participant/session boundary.
    """
    grouped = defaultdict(list)
    names = None
    for header, row in iter_rows(track, split):
        if track == 'fixed':
            # Positional access preserves repeated DU.k.k feature names.
            vals = [float(v) if v else np.nan for v in row[3:]]
            names = [f'{name.strip()}#{i}' for i, name in enumerate(header[3:])]
            grouped[row[0]].append((int(row[2]), vals))
        else:
            vals = [float(v) if v else np.nan for v in row[4:9]]
            grouped[row[0]].append(vals)
    out = {}
    if track == 'fixed':
        for subject, rows in grouped.items():
            out[subject] = np.asarray([v for _, v in sorted(rows)], dtype=float)
    else:
        # 5 robust distribution summaries + mean/std per timing channel.
        # Unlike the reference paper this uses 50/50 disjoint windows, chosen
        # before measuring any result; not the paper's overlapping 30/10.
        names = [f'{stat}.channel{j}' for stat in ['q10', 'q25', 'q50', 'q75', 'q90', 'mean', 'std']
                 for j in range(5)]
        for subject, rows in grouped.items():
            a = np.asarray(rows, float)
            a[~np.isfinite(a)] = np.nan
            features = []
            for start in range(0, len(a) - window + 1, window):
                w = a[start:start + window]
                features.append(np.concatenate([
                    np.nanquantile(w, [.1, .25, .5, .75, .9], axis=0).reshape(-1),
                    np.nanmean(w, axis=0), np.nanstd(w, axis=0)]))
            out[subject] = np.asarray(features).reshape(-1, len(names))
    for a in out.values():
        a[~np.isfinite(a)] = np.nan
    return out, names


def blocks(data):
    """No overlapping observations across fit, selection, and calibration."""
    fit, select, calibrate = {}, {}, {}
    for subject, a in sorted(data.items()):
        n = len(a)
        if n < 12:
            continue
        first, second = int(n * .5), int(n * .75)
        fit[subject], select[subject], calibrate[subject] = a[:first], a[first:second], a[second:]
    return fit, select, calibrate


def flatten(data, subjects):
    return (np.concatenate([data[s] for s in subjects]),
            np.concatenate([np.full(len(data[s]), i, dtype=int) for i, s in enumerate(subjects)]))


class Transform:
    """Train-only imputation, clipping, and standardization."""
    def fit(self, x):
        self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median)
        a = np.where(np.isfinite(x), x, self.median)
        self.lo, self.hi = np.quantile(a, [.005, .995], axis=0)
        a = np.clip(a, self.lo, self.hi)
        self.mean, self.scale = a.mean(0), np.maximum(a.std(0), 1e-6)
        return self

    def apply(self, x):
        a = np.where(np.isfinite(x), x, self.median)
        return (np.clip(a, self.lo, self.hi) - self.mean) / self.scale


class Verifiers:
    def __init__(self, kind, parameter=1.):
        self.kind, self.parameter = kind, parameter

    def fit(self, x, y, n_subjects):
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import ExtraTreesClassifier
        self.transform = Transform().fit(x)
        x = self.transform.apply(x)
        if self.kind == 'logistic':
            self.models = [LogisticRegression(C=self.parameter, solver='liblinear',
                           class_weight='balanced', max_iter=500, random_state=20260920).fit(x, y == k)
                           for k in range(n_subjects)]
        elif self.kind == 'lda':
            self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=self.parameter).fit(x, y)
        elif self.kind == 'extra_trees':
            self.model = ExtraTreesClassifier(n_estimators=128, min_samples_leaf=int(self.parameter),
                max_features='sqrt', class_weight='balanced', n_jobs=2,
                random_state=20260920).fit(x, y)
        else:
            raise ValueError(self.kind)
        return self

    def scores(self, x):
        x = self.transform.apply(x)
        if self.kind == 'logistic':
            return np.column_stack([m.decision_function(x) for m in self.models])
        if self.kind == 'lda':
            return self.model.decision_function(x)
        return self.model.predict_proba(x)


def rates(genuine, impostor, threshold):
    return {'frr': float(np.mean(genuine < threshold)),
            'far': float(np.mean(impostor >= threshold))}


def curve_metrics(genuine, impostor):
    """Tie-aware diagnostic ROC. Threshold here is NEVER deployed."""
    thresholds = np.r_[-np.inf, np.unique(np.r_[genuine, impostor]), np.inf]
    frr = np.searchsorted(np.sort(genuine), thresholds, side='left') / len(genuine)
    far = 1 - np.searchsorted(np.sort(impostor), thresholds, side='left') / len(impostor)
    k = np.argmin(np.abs(frr - far))
    return {'eer': float((frr[k] + far[k]) / 2),
            'oracle_frr_at_far_1pct': float(frr[far <= .01 + 1e-12].min())}


def calibrated_threshold(impostor, target=.01):
    """Largest acceptance region satisfying empirical train-calibration FAR."""
    a = np.sort(impostor)
    allowed = int(np.floor(target * len(a)))
    return float(np.nextafter(a[-allowed - 1], np.inf))


def evaluate(scores, y, subjects, thresholds=None):
    per_user = []
    for k, subject in enumerate(subjects):
        g, i = scores[y == k, k], scores[y != k, k]
        if not len(g) or not len(i):
            continue
        row = {'subject': subject, 'genuine_n': len(g), 'impostor_n': len(i), **curve_metrics(g, i)}
        if thresholds is not None:
            row.update(rates(g, i, thresholds[k]))
        per_user.append(row)
    fields = ['eer', 'oracle_frr_at_far_1pct'] + (['frr', 'far'] if thresholds is not None else [])
    aggregate = {key: float(np.mean([r[key] for r in per_user])) for key in fields}
    rng = np.random.default_rng(20260920)
    # Subject bootstrap reflects user heterogeneity, not independent impostor trials.
    for key in fields:
        a = np.array([r[key] for r in per_user])
        estimates = np.mean(a[rng.integers(0, len(a), size=(1000, len(a)))], axis=1)
        aggregate[key + '_user_bootstrap_95ci'] = np.quantile(estimates, [.025, .975]).tolist()
    return {'aggregate_macro_user': aggregate, 'per_user': per_user}


def run(track, output: Path, window=50):
    from threadpoolctl import threadpool_limits
    import joblib
    import sklearn
    started = time.perf_counter()
    manifest = prepare()
    train, names = load_track(track, 'train', window)
    fit, select, calibration = blocks(train)
    subjects = sorted(fit)
    x, y = flatten(fit, subjects)
    sx, sy = flatten(select, subjects)
    cx, cy = flatten(calibration, subjects)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [('logistic', .1), ('logistic', 1.), ('logistic', 10.),
                  ('lda', .1), ('lda', .5), ('extra_trees', 2), ('extra_trees', 5)]
    selection = []
    models = []
    # All choices above are fixed BEFORE any development features are loaded.
    with threadpool_limits(limits=2):
        for kind, parameter in candidates:
            t = time.perf_counter()
            model = Verifiers(kind, parameter).fit(x, y, len(subjects))
            result = evaluate(model.scores(sx), sy, subjects)
            row = {'kind': kind, 'parameter': parameter,
                   'selection_eer': result['aggregate_macro_user']['eer'],
                   'elapsed_seconds': time.perf_counter() - t}
            selection.append(row)
            models.append(model)
            print(json.dumps({'track': track, **row}), flush=True)
    best = min(range(len(selection)), key=lambda k: (selection[k]['selection_eer'], k))
    lr = min((k for k, (kind, _) in enumerate(candidates) if kind == 'logistic'),
             key=lambda k: selection[k]['selection_eer'])
    # Freeze each candidate's calibration without ever changing its fitting data.
    chosen = {'reference_logistic': lr, 'train_selected': best}
    frozen = {}
    for label, index in chosen.items():
        model = models[index]
        cscore = model.scores(cx)
        thresholds = np.array([calibrated_threshold(cscore[cy != k, k]) for k in range(len(subjects))])
        frozen[label] = {'index': index, 'thresholds': thresholds.tolist(),
                         'calibration': evaluate(cscore, cy, subjects, thresholds)}
        joblib.dump(model, output / f'{track}-{label}.joblib')
    config = {'track': track, 'free_window': window, 'feature_names': names,
              'subjects': subjects, 'candidates': selection, 'frozen': frozen,
              'split_manifest_sha256': checksum(DATA / 'split-manifest.json'),
              'script_sha256': checksum(Path(__file__)), 'sklearn_version': sklearn.__version__,
              'fit_n': len(x), 'selection_n': len(sx), 'calibration_n': len(cx),
              'training_blocks': 'chronological 50% fit / 25% selection / 25% calibration within S1',
              'threshold_rule': 'per-account train-calibration impostor FAR <= 1%; never refitted',
              'test_accessed': False}
    (output / f'{track}-frozen.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / f'{track}-benchmark-source.py').write_text(Path(__file__).read_text())
    # This boundary is deliberate. Development data can only validate a fully
    # frozen configuration; no selection/refitting follows this read.
    return validate_frozen(track, output, config, started)


def validate_frozen(track, output, config=None, started=None):
    """Resume validation from frozen artifacts without refitting or selecting."""
    import joblib
    from threadpoolctl import threadpool_limits
    started = started or time.perf_counter()
    config = config or json.loads((output / f'{track}-frozen.json').read_text())
    if config['split_manifest_sha256'] != checksum(DATA / 'split-manifest.json'):
        raise ValueError('Split changed after fitting')
    subjects, names, frozen = config['subjects'], config['feature_names'], config['frozen']
    dev, dev_names = load_track(track, 'dev', config['free_window'])
    assert dev_names == names
    if not set(subjects).issubset(dev):
        raise ValueError('Missing dev subject; do not silently alter cohort')
    dx, dy = flatten(dev, subjects)
    validation = {}
    with threadpool_limits(limits=2):
        for label in frozen:
            begin = time.perf_counter()
            model = joblib.load(output / f'{track}-{label}.joblib')
            scores = model.scores(dx)
            thresholds = np.asarray(frozen[label]['thresholds'])
            validation[label] = evaluate(scores, dy, subjects, thresholds)
            validation[label]['score_seconds'] = time.perf_counter() - begin
            np.savez_compressed(output / f'{track}-{label}-dev-scores.npz',
                                scores=scores, y=dy, subjects=np.array(subjects), thresholds=thresholds)
            print(json.dumps({'track': track, 'validation': label,
                **validation[label]['aggregate_macro_user']}), flush=True)
    np.savez_compressed(output / f'{track}-dev-features.npz', x=dx, y=dy, subjects=np.array(subjects))
    result = {'dataset': 'KeyRecs 2023', 'track': track, 'evaluation_split': 'dev',
              'development_n': len(dx), 'enrolled_subjects': len(subjects),
              'subjects': len(np.unique(dy)), 'validation': validation,
              'no_complete_dev_window': [s for s in subjects if len(dev[s]) == 0],
              'frozen_config_sha256': checksum(output / f'{track}-frozen.json'),
              'validation_script_sha256': checksum(Path(__file__)),
              'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['No test measurements accessed',
                 'Within-session S1 selection/calibration; S2 validation is cross-session, not proven cross-day',
                 'Fixed uses 47 positional timing values; free uses 35 summaries of 50 nonoverlapping digraphs',
                 'Published method family baseline, not exact paper reproduction or SOTA claim',
                 'Oracle ROC metrics describe dev separation; deployed operating rates use frozen S1 thresholds',
                 'Per-user bootstrap does not make correlated impostor attempts independent']}
    (output / f'{track}-results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def self_check():
    """Synthetic tests of leakage barriers, ties, and temporal separation."""
    import tempfile
    import data_keystrokes as adapter
    assert curve_metrics(np.array([2., 3.]), np.array([0., 1.]))['eer'] == 0
    assert curve_metrics(np.array([1., 1.]), np.array([1., 1.]))['eer'] == .5
    for impostors in [np.arange(100.), np.ones(100), np.array([1., 2.])]:
        threshold = calibrated_threshold(impostors)
        assert np.mean(impostors >= threshold) <= .01
    fitting, selection, cal = blocks({'a': np.arange(100).reshape(20, 5)})
    assert np.array_equal(np.concatenate([fitting['a'], selection['a'], cal['a']]), np.arange(100).reshape(20, 5))
    assert set(fitting['a'][:, 0]).isdisjoint(selection['a'][:, 0])
    assert set(selection['a'][:, 0]).isdisjoint(cal['a'][:, 0])
    original_data = adapter.DATA
    # Only repository-local writes, including temporary synthetic fixtures.
    with tempfile.TemporaryDirectory(prefix='synthetic-keyrecs-', dir=ROOT / 'research/benchmarks') as tmp:
        adapter.DATA = Path(tmp)
        try:
            for track in ['fixed', 'free']:
                path = adapter.DATA / f'{track}-text.csv'
                header = 'participant,session,repetition,value\n' if track == 'fixed' else 'participant,session,key1,key2,a,b,c,d,e,\n'
                valid = 'p002,1,1,0.2\n' if track == 'fixed' else 'p002,1,2,",0.1,0.2,0.3,0.4,0.5,\n'
                path.write_text(header + 'p001,1,SEALED_POISON\np001,2,SEALED_POISON\n' + valid)
            files = {f'{t}-text.csv': {'sha256': checksum(adapter.DATA / f'{t}-text.csv')} for t in ['fixed', 'free']}
            (adapter.DATA / 'split-manifest.json').write_text(json.dumps({'active_subjects': ['p002'],
                 'sealed_test_subjects': ['p001'], 'files': files}))
            assert len(list(adapter.iter_rows('fixed', 'train'))) == 1
            rows = list(adapter.iter_rows('free', 'train'))
            assert len(rows) == 1 and list(map(float, rows[0][1][4:9])) == [.1, .2, .3, .4, .5]
            try:
                list(adapter.iter_rows('fixed', 'test'))
                raise AssertionError('Test access was permitted')
            except ValueError:
                pass
            path = adapter.DATA / 'fixed-text.csv'
            path.write_text(path.read_text() + 'p002,1,2,0.3\n')
            try:
                adapter.prepare()
                raise AssertionError('Modified input was permitted')
            except ValueError:
                pass
        finally:
            adapter.DATA = original_data
    print('Passed: sealed poison skipped, test access denied, source tamper blocked, malformed quote keys parsed, ties conservative, blocks disjoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=['fixed', 'free', 'both'], default='both')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/benchmarks/results/keyrecs-v1')
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--validate-frozen', action='store_true', help='Resume existing frozen models; no training')
    parser.add_argument('--self-check', action='store_true', help='Synthetic integrity tests; no dataset values accessed')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.window < 10:
        parser.error('window must contain at least 10 digraphs')
    for track in (['fixed', 'free'] if args.track == 'both' else [args.track]):
        if (args.output / f'{track}-results.json').exists():
            raise SystemExit('Results already exist; choose a new output directory and log your new experiment')
        if args.validate_frozen:
            validate_frozen(track, args.output)
        else:
            run(track, args.output, args.window)


if __name__ == '__main__':
    # Model bundles must load from another process, not depend on the caller's
    # __main__ namespace when this file was invoked as a script.
    sys.modules['keystroke_benchmark'] = sys.modules[__name__]
    Transform.__module__ = 'keystroke_benchmark'
    Verifiers.__module__ = 'keystroke_benchmark'
    main()
