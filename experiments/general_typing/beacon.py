"""Frozen, keyboard-only gameplay transfer diagnostic; see BEACON_PROTOCOL.md."""
import csv
import json
from collections import defaultdict
import numpy as np
import joblib
from run import ROOT, OUT, checksum, summarize, profile, comparison, score, metrics


def main():
    destination = OUT / 'beacon.json'
    if destination.exists():
        raise FileExistsError(destination)
    base = ROOT / 'datasets/beacon'
    manifest = json.loads((base / 'split_manifest.json').read_text())
    ledger = json.loads((base / 'record_roles.json').read_text())
    assert ledger['manifest_sha256'] == checksum(base / 'split_manifest.json')
    roles = {r['release_path']: r for r in ledger['files']}
    (OUT / 'beacon-plan.json').write_text(json.dumps({
        'protocol_sha256': checksum(__import__('pathlib').Path(__file__).with_name('BEACON_PROTOCOL.md')),
        'manifest_sha256': ledger['manifest_sha256'], 'script_sha256': checksum(__import__('pathlib').Path(__file__)),
        'frozen_models_sha256': checksum(OUT / 'frozen.json')}, indent=2))
    grouped = defaultdict(dict)
    audit = []
    for row in manifest['files']:
        if row['split'] != 'dev' or row['modality'] != 'keyboard_csv':
            continue
        entry = roles[row['release_path']]
        expected = 'train' if row['role'] == 'enrollment' else 'dev'
        if entry['record_partition'] != expected:
            raise ValueError('Unexpected record role')
        path = base / 'dev' / row['release_path']
        if checksum(path) != row['sha256']:
            raise ValueError('Source checksum mismatch')
        with path.open() as stream:
            events = np.array([(float(r['Elapsed Start Time']), float(r['Elapsed Release Time'])) for r in csv.DictReader(stream)]) * 1000
        events = events[np.argsort(events[:, 0], kind='stable')]
        vectors, rejected = [], 0
        for start in range(0, len(events) - 9, 10):
            window = events[start:start + 10]
            try:
                vectors.append(summarize(window[:, 1] - window[:, 0], np.diff(window[:, 0]),
                                         window[1:, 0] - window[:-1, 1]))
            except ValueError:
                rejected += 1
        grouped[row['participant_id']][row['role']] = np.asarray(vectors)
        audit.append({'subject': row['participant_id'], 'role': row['role'], 'valid_windows': len(vectors),
                      'rejected_windows': rejected, 'tail_events': len(events) % 10})
    subjects = sorted(s for s, data in grouped.items() if len(data['enrollment']) >= 10 and len(data['probe']))
    if len(subjects) < 2:
        raise ValueError('Insufficient eligible identities')
    queries = np.concatenate([grouped[s]['probe'] for s in subjects])
    actual = np.concatenate([np.full(len(grouped[s]['probe']), i) for i, s in enumerate(subjects)])
    x = np.concatenate([comparison(profile(grouped[s]['enrollment'][:10]), queries) for s in subjects])
    y = np.concatenate([actual == i for i in range(len(subjects))])
    results = {}
    for key, entry in json.loads((OUT / 'frozen.json').read_text()).items():
        path = OUT / entry['file']
        if checksum(path) != entry['sha256']:
            raise ValueError('Model changed')
        artifact = joblib.load(path)
        scores = score(artifact['model'], x)
        results[key] = metrics(y, scores, artifact['threshold'])
        np.savez_compressed(OUT / f'beacon-{key.replace("/", "-")}-scores.npz', scores=scores, genuine=y)
    destination.write_text(json.dumps({'subjects': subjects, 'audit': audit, 'results': results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
