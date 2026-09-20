"""Conditional paired probe-person bootstrap of already frozen DEV scores.

No fitting or model selection. Enrolled reference identities and models remain
fixed; intervals do not cover reference, training, or prior DEV reuse uncertainty.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/benchmarks/pointer_sapimouse_znorm_dev_v2'
OUT = ROOT / 'research/benchmarks/pointer_znorm_uncertainty_v1'
METHODS = {'cosine@5': 'cosine', 'znorm_cosine@5': 'znorm'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def weighted_rates(labels, scores, weights, threshold):
    labels, scores, weights = map(np.asarray, (labels, scores, weights))
    if labels.ndim != 1 or labels.shape != scores.shape or labels.shape != weights.shape:
        raise ValueError('Expected equal one-dimensional arrays')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid labels/scores')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid weights')
    positive, negative = labels == 1, labels == 0
    if weights[positive].sum() <= 0 or weights[negative].sum() <= 0:
        raise ValueError('Both classes require positive weight')
    far = weights[negative & (scores >= threshold)].sum() / weights[negative].sum()
    frr = weights[positive & (scores < threshold)].sum() / weights[positive].sum()
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    index = np.argmin(np.abs(fpr - (1 - tpr)))
    return np.array([far, frr, (fpr[index] + 1 - tpr[index]) / 2])


def prepare():
    OUT.mkdir(exist_ok=False)
    plan = dict(seed=20260920, replicates=2000, target='0.01',
                cluster='probe person; all decisions and claimed accounts retained together',
                interval='paired percentile 2.5/97.5%; candidate minus selected',
                limitations='Conditional on fixed enrollment references and models; shared-reference, training and prior DEV reuse uncertainty not represented.',
                evaluation_only=True, candidate_promoted=False,
                inputs={name: digest(SOURCE / name) for name in ['dev_complete.json', 'scores.npz']},
                source_sha256=digest(Path(__file__)))
    write(OUT / 'plan.json', plan)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())


def run():
    plan = json.loads((OUT / 'plan.json').read_text())
    if digest(Path(__file__)) != plan['source_sha256']:
        raise ValueError('Source changed after preparation')
    for name, expected in plan['inputs'].items():
        if digest(SOURCE / name) != expected:
            raise ValueError('Frozen input changed')
    report = json.loads((SOURCE / 'dev_complete.json').read_text())
    if report['scores_sha256'] != plan['inputs']['scores.npz']:
        raise ValueError('Report/score binding failed')
    with np.load(SOURCE / 'scores.npz', allow_pickle=False) as data:
        accounts, actual = data['accounts'], data['actual']
        identities, cluster = np.unique(actual, return_inverse=True)
        if len(identities) != 24 or set(identities) != set(accounts):
            raise ValueError('Expected original full DEV cohort')
        labels = (accounts[:, None] == actual[None, :]).astype(int).ravel()
        rng = np.random.default_rng(plan['seed'])
        draws = rng.multinomial(len(identities), np.full(len(identities), 1 / len(identities)), size=plan['replicates'])
        values = {}
        observed = {}
        for method, key in METHODS.items():
            matrix = data[key]
            if matrix.shape != (len(accounts), len(actual)):
                raise ValueError('Unexpected score axes')
            scores = matrix.ravel()
            threshold = report['plan']['thresholds'][method][plan['target']]
            point = weighted_rates(labels, scores, np.ones(len(scores)), threshold)
            expected = report['dev'][method][plan['target']]
            np.testing.assert_allclose(point, [expected[k] for k in ['far', 'frr', 'eer_descriptive']], atol=1e-14, rtol=0)
            observed[method] = point.tolist()
            values[method] = np.array([weighted_rates(labels, scores,
                np.tile(draw[cluster], len(accounts)), threshold) for draw in draws])
    delta = values['znorm_cosine@5'] - values['cosine@5']
    output = dict(plan=plan, clusters=len(identities), decisions=len(actual),
                  metric_order=['far', 'frr', 'eer_descriptive'], observed=observed,
                  conditional_95pct={name: np.quantile(value, [.025, .975], axis=0).T.tolist()
                                     for name, value in values.items()},
                  candidate_minus_selected_95pct=np.quantile(delta, [.025, .975], axis=0).T.tolist(),
                  test_accessed=False, candidate_promoted=False)
    write(OUT / 'results.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
