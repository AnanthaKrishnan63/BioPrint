"""Audit pointer file-role metadata without opening any test measurements."""
import collections
import json
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
reports = {}
for dataset in ['balabit', 'sapimouse']:
    base = ROOT / 'data/benchmarks' / dataset
    manifest = json.loads((base / 'split_manifest.json').read_text())
    entries = manifest.get('entries', manifest.get('files'))
    paths = [e['path'] for e in entries]
    assert len(paths) == len(set(paths)), 'A file has more than one split role'
    for entry in entries:
        path = base / entry['path']
        if entry['split'] == 'test_sealed':
            assert not path.exists(), f'Sealed content extracted: {entry["path"]}'
        else:
            assert path.exists(), f'Missing permitted file: {entry["path"]}'
    if dataset == 'sapimouse':
        support = [e for e in entries if e['role'] == 'dev_support_enrollment']
        probe = [e for e in entries if e['role'] == 'dev_probe']
        assert len(support) == len(probe) == 24
        assert all(e['split'] == 'train' and e['path'].endswith('_3min.csv') for e in support)
        assert all(e['split'] == 'dev' and e['path'].endswith('_1min.csv') for e in probe)
        represented = {e['path'].split('/')[1] for e in entries if e['role'] == 'representation'}
        evaluation = {e['path'].split('/')[1] for e in support}
        sealed = {e['path'].split('/')[1] for e in entries if e['split'] == 'test_sealed'}
        assert not represented & evaluation and not represented & sealed and not evaluation & sealed
    reports[dataset] = {'files': len(entries), 'by_split': dict(collections.Counter(e['split'] for e in entries)), 'sealed_content_present_on_filesystem': False}
print(json.dumps(reports, indent=2))
