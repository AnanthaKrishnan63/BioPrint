"""Bounded exploratory follow-up; see ALIGNED_PROTOCOL.md."""
import json
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run import role, checksum, calibrated_threshold, metrics, SEED, BRANCH
from control import raw
from engine import scorer
from engine.aligned_typing import comparison

OUT = Path(__file__).parent / 'aligned_results'


def trials(track, train, names, probes, stage):
    namespace = 'cmu' if track == 'cmu' else 'keyrecs'
    subjects = sorted(s for s in train if len(train[s]) >= 11 and s in probes and len(probes[s])
                      and role(namespace, s) == stage)
    values = {s: probes[s][10:] if stage != 'evaluation' else probes[s] for s in subjects}
    x = np.concatenate([values[s] for s in subjects])
    actual = np.concatenate([np.full(len(values[s]), i) for i, s in enumerate(subjects)])
    profiles = {s: scorer.fit(train[s][:10].tolist(), names) for s in subjects}
    return (np.concatenate([comparison(profiles[s], x) for s in subjects]),
            np.concatenate([actual == i for i in range(len(subjects))]), subjects)


def score(model, x):
    if model is None:
        return -x[:, -1]
    return model.decision_function(x) if hasattr(model, 'decision_function') else model.predict_proba(x)[:, 1]


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps({
        'protocol_sha256': checksum(Path(__file__).with_name('ALIGNED_PROTOCOL.md')),
        'script_sha256': checksum(Path(__file__)),
        'engine_sha256': checksum(BRANCH / 'code/bioprint/engine/aligned_typing.py'),
        'prior_summary_dev_exposure': True, 'sealed_test_read': False}, indent=2))
    train = {t: raw(t, 'train') for t in ('cmu', 'fixed')}
    fit = {t: trials(t, x, names, x, 'fit') for t, (names, x) in train.items()}
    cal = {t: trials(t, x, names, x, 'calibration') for t, (names, x) in train.items()}
    frozen = {}
    for protocol, sources in {'pooled': ['cmu', 'fixed'], 'cmu_only': ['cmu'], 'keyrecs_only': ['fixed']}.items():
        x, y = (np.concatenate([fit[t][i] for t in sources]) for i in (0, 1))
        weights = np.concatenate([np.where(fit[t][1], .5 / fit[t][1].sum(), .5 / (~fit[t][1]).sum()) for t in sources])
        weights *= len(weights) / weights.sum()
        cx, cy = (np.concatenate([cal[t][i] for t in sources]) for i in (0, 1))
        models = {'distance': None,
                  'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED)),
                  'extra_trees': ExtraTreesClassifier(n_estimators=128, min_samples_leaf=10, n_jobs=2, random_state=SEED)}
        for name, model in models.items():
            if model is not None:
                weights_arg = {'logisticregression__sample_weight': weights} if name == 'logistic' else {'sample_weight': weights}
                model.fit(x, y, **weights_arg)
            cs = score(model, cx)
            threshold = calibrated_threshold(cs[~cy])
            path = OUT / f'{protocol}-{name}.joblib'
            joblib.dump({'schema': 'aligned-personal-residuals-v1', 'model': model, 'threshold': threshold}, path)
            frozen[f'{protocol}/{name}'] = {'file': path.name, 'sha256': checksum(path),
                                           'sources': sources, 'calibration': metrics(cy, cs, threshold)}
            print(f'Frozen aligned {protocol}/{name}', flush=True)
    (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    results = {}
    for track, (names, rows) in train.items():
        _, dev = raw(track, 'dev')
        x, y, subjects = trials(track, rows, names, dev, 'evaluation')
        for key, entry in frozen.items():
            artifact = joblib.load(OUT / entry['file'])
            scores = score(artifact['model'], x)
            result = metrics(y, scores, artifact['threshold'])
            result.update({'subjects': subjects, 'dataset_transfer': track not in entry['sources']})
            results[f'{track}/{key}'] = result
            np.savez_compressed(OUT / f'{track}-{key.replace("/", "-")}-scores.npz', scores=scores, genuine=y)
            print(f'{track}/{key}: FAR={result["far"]:.4f} FRR={result["frr"]:.4f} EER={result["eer"]:.4f}', flush=True)
    (OUT / 'report.json').write_text(json.dumps({'results': results, 'seconds': time.perf_counter() - started,
        'fit_profiles': {t: fit[t][2] for t in fit}, 'calibration_profiles': {t: cal[t][2] for t in cal}}, indent=2))


if __name__ == '__main__':
    main()
