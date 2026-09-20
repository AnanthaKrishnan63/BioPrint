"""Run declared, isolated TRAIN/calibration/DEV general typing experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

BRANCH = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ['BIOPRINT_RESEARCH_ROOT']).resolve()
sys.path.insert(0, str(BRANCH / 'code/bioprint'))
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from engine.general_typing import summarize, profile, comparison
from data_keystrokes import iter_rows, prepare, checksum, DATA
from keystroke_benchmark import calibrated_threshold, curve_metrics
# Original guarded CMU loader lives in the research workspace.
import importlib.util
spec = importlib.util.spec_from_file_location('guarded_cmu', ROOT / 'code/bioprint/eval/strict_cmu.py')
cmu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmu)

OUT = Path(__file__).parent / 'results'
SEED = 20260920
AUDIT = {}


def role(namespace, subject):
    # Namespace groups fixed/free recordings of the same KeyRecs person.
    bucket = int(hashlib.sha256(f'general-v1:{namespace}:{subject}'.encode()).hexdigest()[:8], 16) % 4
    return 'fit' if bucket < 2 else 'calibration' if bucket == 2 else 'evaluation'


def load(track, split):
    grouped = defaultdict(list)
    rejected = 0
    tails = 0
    if track == 'cmu':
        names, source = cmu.load_partition(split)
        indices = [[i for i, n in enumerate(names) if n.startswith(prefix)]
                   for prefix in ('H.', 'DD.', 'UD.')]
        candidates = ((s, [x[idx] for idx in indices]) for s, rows in source.items() for x in rows['X'])
    elif track == 'fixed':
        def fixed():
            for header, row in iter_rows('fixed', split):
                indices = [[], [], []]
                for i, name in enumerate(header[3:]):
                    parts = name.strip().split('.')
                    if len(parts) == 3 and parts[0] == 'DU' and parts[1] == parts[2]:
                        indices[0].append(i)
                    elif name.startswith('DD.'):
                        indices[1].append(i)
                    elif name.startswith('UD.'):
                        indices[2].append(i)
                values = np.asarray([float(v) if v else np.nan for v in row[3:]]) * 1000
                yield row[0], [values[idx] for idx in indices]
        candidates = fixed()
    else:
        raw = defaultdict(list)
        for _, row in iter_rows('free', split):
            raw[row[0]].append([float(row[i]) * 1000 if row[i] else np.nan for i in (4, 5, 7)])
        tails = sum(len(x) % 9 for x in raw.values())
        candidates = ((s, np.asarray(rows[start:start + 9]).T)
                      for s, rows in raw.items() for start in range(0, len(rows) - 8, 9))
    for subject, channels in candidates:
        try:
            grouped[subject].append(summarize(*channels))
        except ValueError:
            rejected += 1
    AUDIT[f'{track}/{split}'] = {'rejected_vectors': rejected, 'discarded_tail_digraphs': tails,
                                 'valid_vectors': sum(map(len, grouped.values()))}
    return {s: np.asarray(x) for s, x in grouped.items()}


def trials(track, train, probes, requested_role, training=False):
    namespace = 'cmu' if track == 'cmu' else 'keyrecs'
    subjects = sorted(s for s in train if len(train[s]) >= 11 and s in probes
                      and len(probes[s]) and role(namespace, s) == requested_role)
    if len(subjects) < 2:
        raise ValueError(f'Insufficient subjects: {track}/{requested_role}')
    query = {s: probes[s][10:] if training else probes[s] for s in subjects}
    values = np.concatenate([query[s] for s in subjects])
    actual = np.concatenate([np.full(len(query[s]), i) for i, s in enumerate(subjects)])
    x, y, owner = [], [], []
    for i, s in enumerate(subjects):
        x.append(comparison(profile(train[s][:10]), values))
        y.append(actual == i)
        owner.extend([s] * len(values))
    return np.concatenate(x), np.concatenate(y), np.asarray(owner), subjects


def metrics(y, scores, threshold):
    g, i = scores[y], scores[~y]
    fa, fr = int((i >= threshold).sum()), int((g < threshold).sum())
    return {'false_accepts': fa, 'impostor_attempts': len(i), 'false_rejects': fr,
            'genuine_attempts': len(g), 'far': fa / len(i), 'frr': fr / len(g),
            'eer': curve_metrics(g, i)['eer'], 'threshold': float(threshold)}


def score(model, x):
    if model is None:
        return -np.minimum(x[:, :21], 6).mean(axis=1)
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    manifest = prepare()
    plan = {'protocol_sha256': checksum(Path(__file__).with_name('PROTOCOL.md')),
            'script_sha256': checksum(Path(__file__)),
            'engine_sha256': checksum(BRANCH / 'code/bioprint/engine/general_typing.py'),
            'keyrecs_manifest_sha256': checksum(DATA / 'split-manifest.json'),
            'cmu_source_sha256': checksum(cmu.SOURCE),
            'sealed_test_observations_decoded': False, 'prior_dev_exposure': True,
            'roles': {'keyrecs': {s: role('keyrecs', s) for s in manifest['active_subjects']}}}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    tracks = ['cmu', 'fixed', 'free']
    train = {t: load(t, 'train') for t in tracks}
    plan['roles']['cmu'] = {s: role('cmu', s) for s in train['cmu']}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    fit = {t: trials(t, train[t], train[t], 'fit', True) for t in tracks}
    cal = {t: trials(t, train[t], train[t], 'calibration', True) for t in tracks}
    protocols = {'pooled': tracks, 'cmu_only': ['cmu'], 'keyrecs_only': ['fixed', 'free']}
    frozen = {}
    for protocol, sources in protocols.items():
        x = np.concatenate([fit[t][0] for t in sources])
        y = np.concatenate([fit[t][1] for t in sources])
        weights = np.concatenate([np.where(fit[t][1], .5 / fit[t][1].sum(),
                                           .5 / (~fit[t][1]).sum()) for t in sources])
        weights *= len(weights) / weights.sum()
        cx = np.concatenate([cal[t][0] for t in sources])
        cy = np.concatenate([cal[t][1] for t in sources])
        models = {'distance': None,
                  'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED)),
                  'extra_trees': ExtraTreesClassifier(n_estimators=128, min_samples_leaf=10,
                                                       n_jobs=2, random_state=SEED)}
        for name, model in models.items():
            if model is not None:
                kwargs = {'logisticregression__sample_weight': weights} if name == 'logistic' else {'sample_weight': weights}
                model.fit(x, y, **kwargs)
            scores = score(model, cx)
            threshold = calibrated_threshold(scores[~cy], .01)
            path = OUT / f'{protocol}-{name}.joblib'
            joblib.dump({'model': model, 'threshold': threshold, 'schema': 'general-typing-summary-v1'}, path)
            frozen[f'{protocol}/{name}'] = {'sources': sources, 'file': path.name,
                'sha256': checksum(path), 'calibration': metrics(cy, scores, threshold)}
            print(f'Frozen {protocol}/{name}', flush=True)
    (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    # All nine model/threshold combinations are frozen before the first DEV read.
    results = {}
    for track in tracks:
        dev = load(track, 'dev')
        x, y, owners, subjects = trials(track, train[track], dev, 'evaluation')
        for key, item in frozen.items():
            saved = joblib.load(OUT / item['file'])
            scores = score(saved['model'], x)
            row = metrics(y, scores, saved['threshold'])
            row.update({'subjects': subjects, 'subject_count': len(subjects),
                        'dataset_transfer': track not in item['sources'],
                        'per_user': {s: metrics(y[owners == s], scores[owners == s], saved['threshold']) for s in subjects}})
            results[f'{track}/{key}'] = row
            np.savez_compressed(OUT / f'{track}-{key.replace("/", "-")}-scores.npz',
                                scores=scores, genuine=y, claimed=owners)
            print(f'{track}/{key}: FAR={row["far"]:.4f} FRR={row["frr"]:.4f} EER={row["eer"]:.4f}', flush=True)
    report = {'results': results, 'audit': AUDIT, 'seconds': time.perf_counter() - started,
              'fit_profiles': {t: fit[t][3] for t in tracks}, 'calibration_profiles': {t: cal[t][3] for t in tracks}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
