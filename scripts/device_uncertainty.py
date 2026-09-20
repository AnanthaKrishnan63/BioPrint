"""Probe-browser cluster bootstrap for frozen device dev operating points.

Intervals condition on enrolled references/trained models. Shared impostor
references introduce additional dependence not captured by this one-way bootstrap.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks'


def main():
    report = json.loads((OUT / 'device_results.json').read_text())
    with np.load(OUT / 'device_dev_scores.npz', allow_pickle=False) as data:
        labels, owners = data['labels'], data['owners']
        identities, cluster = np.unique(owners, return_inverse=True)
        rng = np.random.default_rng(20260920)
        multiplicities = rng.multinomial(len(identities), np.full(len(identities), 1 / len(identities)), size=2000)
        positive, negative = labels == 1, labels == 0
        denominators = [multiplicities @ np.bincount(cluster[mask], minlength=len(identities)) for mask in [negative, positive]]
        output = {'replicates': 2000, 'cluster': 'probe browser identity', 'clusters': len(identities),
                  'limitations': 'Conditional one-way intervals; references/model uncertainty and shared-reference dependence are not resampled.',
                  'methods': {}}
        for name, method in report['methods'].items():
            output['methods'][name] = {}
            for target, values in method['operating_points'].items():
                threshold = values['threshold_selected_on_training_calibration']
                accept = data[name] >= threshold
                ci = {}
                for rate, mask, errors, denom in [('far', negative, accept, denominators[0]),
                                                  ('frr', positive, ~accept, denominators[1])]:
                    error_counts = np.bincount(cluster[mask], weights=errors[mask], minlength=len(identities))
                    samples = (multiplicities @ error_counts) / denom
                    ci[rate + '_conditional_cluster_95pct'] = np.quantile(samples, [.025, .975]).tolist()
                output['methods'][name][target] = ci
    (OUT / 'device_uncertainty.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'clusters': len(identities), 'replicates': 2000}))


if __name__ == '__main__':
    main()
