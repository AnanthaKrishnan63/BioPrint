"""Participant deletion sensitivity of frozen cognitive dev operating points.

Remove each participant as both probe and enrolled reference. These are
sensitivity ranges, not confidence intervals or independent pair estimates.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
import joblib
import numpy as np


def main():
    out = ROOT / 'research/benchmarks'
    report = json.loads((out / 'cognitive_results.json').read_text())
    schema = json.loads((out / 'cognitive/schema.json').read_text())
    data = np.load(out / 'cognitive/dev_features.npz', allow_pickle=False)
    n = len(report['dev_subjects'])
    # pairs() orders enrolled owner outermost, probe identity innermost.
    enrolled, probes = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    labels = data['y']
    assert np.array_equal(labels, (enrolled == probes).astype(int))
    differences = data['signed_difference']
    means = differences[:, [0, 5, 10, 15, 20, 25]]
    representations = {
        'condition_index_slope': (means[:, [2, 5]] - means[:, [0, 3]]) / 2,
        'condition_means': means,
        'full_rt_accuracy_profile': differences,
    }
    scores = {}
    for name, values in representations.items():
        model = schema['distance_models'][name]
        scores[name] = -np.sqrt(np.mean(
            (values - np.array(model['drift'])) ** 2 / np.array(model['variance']), axis=1))
    model = joblib.load(out / 'cognitive/learned_full_profile_rf.joblib')
    scores['learned_full_profile_rf'] = model.predict_proba(data['X'])[:, 1]
    output = {
        'participants': n, 'genuine_comparisons': n, 'impostor_comparisons': n * (n - 1),
        'method': 'Delete each participant from both enrollment references and probes, hold model and training-calibrated threshold fixed.',
        'interpretation': 'Sensitivity ranges, NOT confidence intervals. Nine participants cannot establish low-FAR performance. Shared participants make 72 impostor comparisons dependent; training/model uncertainty is not included.',
        'methods': {},
    }
    for name, values in scores.items():
        threshold = report['methods'][name]['operating_points']['0.01']['threshold_selected_on_training_calibration']
        accepts = values >= threshold
        rows = []
        for removed in range(n):
            keep = (enrolled != removed) & (probes != removed)
            rows.append({'removed_subject': report['dev_subjects'][removed],
                         'far': float(np.mean(accepts[keep & (labels == 0)])),
                         'frr': float(np.mean(~accepts[keep & (labels == 1)]))})
        output['methods'][name] = {'deletions': rows,
            'far_range': [min(r['far'] for r in rows), max(r['far'] for r in rows)],
            'frr_range': [min(r['frr'] for r in rows), max(r['frr'] for r in rows)]}
    (out / 'cognitive_uncertainty.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'deletions'}
                      for name, result in output['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
