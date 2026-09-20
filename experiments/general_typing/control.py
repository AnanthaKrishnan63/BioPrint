"""Existing positional scorer on the identical eligible fixed-text DEV trials.

This is a diagnostic comparator, not a source of threshold/model selection.
Free text has no fixed positional password schema, so no false control claim.
"""
import json
from collections import defaultdict
import numpy as np
from run import cmu, iter_rows, summarize, role, OUT, metrics
from engine import scorer


def raw(track, split):
    if track == 'cmu':
        names, rows = cmu.load_partition(split)
        return names, {s: r['X'] for s, r in rows.items()}
    grouped = defaultdict(list)
    for header, row in iter_rows('fixed', split):
        indices, names, channels = [], [], [[], [], []]
        for i, label in enumerate(header[3:]):
            p = label.strip().split('.')
            family = ('H' if len(p) == 3 and p[0] == 'DU' and p[1] == p[2]
                      else p[0] if p[0] in ('DD', 'UD') else None)
            if family:
                indices.append(i)
                names.append(f'{family}.position{i}')
                channels[('H', 'DD', 'UD').index(family)].append(len(indices) - 1)
        x = np.asarray([float(row[3 + i]) * 1000 if row[3 + i] else np.nan for i in indices])
        try:
            summarize(*(x[idx] for idx in channels))
        except ValueError:
            continue
        grouped[row[0]].append(x)
    return names, {s: np.asarray(x) for s, x in grouped.items()}


def main():
    report = json.loads((OUT / 'report.json').read_text())
    result = {}
    for track in ('cmu', 'fixed'):
        names, train = raw(track, 'train')
        _, dev = raw(track, 'dev')
        subjects = report['results'][f'{track}/pooled/distance']['subjects']
        x = np.concatenate([dev[s] for s in subjects])
        actual = np.concatenate([np.full(len(dev[s]), i) for i, s in enumerate(subjects)])
        scores, labels = [], []
        for i, s in enumerate(subjects):
            model = scorer.fit(train[s][:10].tolist(), names)
            d = np.minimum(abs(x - model.center) / model.spread, model.cap).mean(axis=1)
            # Ratio allows one operating threshold while preserving each profile's limit.
            scores.append(-d / model.threshold)
            labels.append(actual == i)
            np.testing.assert_allclose(d[0], scorer.distance(model, x[0])[0])
        y, s = np.concatenate(labels), np.concatenate(scores)
        row = metrics(y, s, -1.)
        for key in ('genuine_attempts', 'impostor_attempts'):
            assert row[key] == report['results'][f'{track}/pooled/distance'][key]
        result[track] = row
    with (OUT / 'control.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
