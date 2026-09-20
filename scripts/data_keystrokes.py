"""Acquire KeyRecs and seal identity holdouts BEFORE loading timing values.

Run with the bigidea Python environment. No command exposes test measurements.
The original CSVs remain untouched; a manifest records the metadata-only split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/benchmarks/data/keyrecs'
SOURCE = 'https://zenodo.org/api/records/7886743'
MAX_FILE = 499_999_999
MAX_TOTAL = 5_000_000_000
SALT = 'bioprint-keyrecs-v1-20260920'


def download(url: str, path: Path, expected_size: int | None = None) -> None:
    """Bound network writes; never leave a completed-looking partial file."""
    if expected_size is not None and expected_size > MAX_FILE:
        raise ValueError('Dataset exceeds the per-file size budget')
    # Counting the entire research tree is deliberately more conservative than
    # counting only recognized datasets. Existing CMU data lives elsewhere.
    used = sum(p.stat().st_size for p in (ROOT / 'research').rglob('*') if p.is_file())
    used += sum(p.stat().st_size for p in (ROOT / 'code/bioprint/eval/data').rglob('*') if p.is_file())
    if used + (expected_size or MAX_FILE) > MAX_TOTAL:
        raise ValueError('Project dataset budget would be exceeded')
    tmp = path.with_suffix(path.suffix + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as src, tmp.open('wb') as out:
            while chunk := src.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE or used + size > MAX_TOTAL:
                    raise ValueError('Download exceeded size limit')
                out.write(chunk)
        if expected_size is not None and size != expected_size:
            raise ValueError('Download size differs from source metadata')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def checksum(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as src:
        while chunk := src.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def acquire() -> None:
    meta_path = DATA / 'source-metadata.json'
    if not meta_path.exists():
        download(SOURCE, meta_path)
    metadata = json.loads(meta_path.read_text())
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Source license changed; review before downloading')
    for item in metadata['files']:
        if item['key'] not in ('fixed-text.csv', 'free-text.csv'):
            continue
        path = DATA / item['key']
        if not path.exists():
            download(item['links']['self'], path, item['size'])
        alg, expected = item['checksum'].split(':')
        if path.stat().st_size != item['size'] or checksum(path, alg) != expected:
            raise ValueError(f'Source integrity check failed: {path.name}')


def metadata_only(path: Path) -> Counter:
    """Inspect only the first TWO CSV fields; remainder is never parsed."""
    counts = Counter()
    with path.open(newline='') as src:
        header = next(csv.reader([next(src)]))
        if header[:2] != ['participant', 'session']:
            raise ValueError('Unexpected metadata prefix')
        for line in src:
            participant, session, _opaque = line.split(',', 2)
            if not participant.startswith('p') or not participant[1:].isdigit():
                raise ValueError('Unexpected participant metadata')
            if session not in ('1', '2'):
                raise ValueError('Unexpected acquisition session')
            counts[(participant, session)] += 1
    return counts


def prepare() -> dict:
    """Commit split based on identifiers only, without numerical timing reads."""
    manifest_path = DATA / 'split-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, spec in manifest['files'].items():
            if checksum(DATA / name) != spec['sha256']:
                raise ValueError('Input changed after split was sealed')
        return manifest
    counts = {name: metadata_only(DATA / name) for name in ('fixed-text.csv', 'free-text.csv')}
    subjects = sorted({p for table in counts.values() for p, _ in table})
    order = sorted(subjects, key=lambda p: hashlib.sha256(f'{SALT}:{p}'.encode()).hexdigest())
    sealed = sorted(order[:max(1, round(len(subjects) * .2))])
    active = sorted(set(subjects) - set(sealed))
    manifest = {
        'version': 1, 'source': SOURCE, 'license': 'CC-BY-4.0', 'salt': SALT,
        'train_rule': 'non-test subjects, acquisition session 1',
        'dev_rule': 'same non-test subjects, acquisition session 2',
        'test_rule': 'all records from reserved subject IDs; never parse timing values',
        'selection_rule': 'training-only chronological blocks; dev for frozen validation only',
        'limitations': ['two sessions do not establish cross-day generalization',
                       'test subjects need an enrollment protocol if later explicitly unsealed',
                       'free-text means prompted transcription, not spontaneous composition'],
        'active_subjects': active, 'sealed_test_subjects': sealed,
        'files': {},
    }
    for name, table in counts.items():
        manifest['files'][name] = {
            'size_bytes': (DATA / name).stat().st_size,
            'sha256': checksum(DATA / name),
            'rows': {
                'train': sum(n for (p, s), n in table.items() if p in active and s == '1'),
                'dev': sum(n for (p, s), n in table.items() if p in active and s == '2'),
                'sealed_test': sum(n for (p, _), n in table.items() if p in sealed),
            },
        }
    # Exclusive creation prevents accidental replacement with a new split.
    with manifest_path.open('x') as out:
        json.dump(manifest, out, indent=2)
        out.write('\n')
    return manifest


def iter_rows(track: str, split: str):
    """Yield train/dev rows only. Test timing strings never reach csv parsing."""
    if track not in ('fixed', 'free') or split not in ('train', 'dev'):
        raise ValueError('Only fixed/free train/dev reads are authorized')
    manifest = prepare()
    active = set(manifest['active_subjects'])
    session = '1' if split == 'train' else '2'
    with (DATA / f'{track}-text.csv').open(newline='') as src:
        header = next(csv.reader([next(src)]))
        for line in src:
            participant, row_session, _opaque = line.split(',', 2)
            if participant not in active or row_session != session:
                continue
            if track == 'free':
                # Eighteen training rows contain unescaped literal quote keys.
                # Timing fields themselves are valid; parse the five numeric
                # suffix columns from the right without interpreting key text.
                # This branch executes only AFTER the split/identity guard.
                suffix = line.rstrip('\r\n')
                if suffix.endswith(','):
                    suffix = suffix[:-1]
                parts = suffix.rsplit(',', 5)
                if len(parts) != 6:
                    raise ValueError('Unexpected free-text timing suffix')
                yield header, [participant, row_session, '', '', *parts[1:], '']
            else:
                yield header, next(csv.reader([line]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        acquire()
    manifest = prepare()
    print(json.dumps({'active_subjects': len(manifest['active_subjects']),
                      'sealed_test_subjects': len(manifest['sealed_test_subjects']),
                      'files': manifest['files']}, indent=2))


if __name__ == '__main__':
    main()
