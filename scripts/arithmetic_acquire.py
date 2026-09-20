"""Freeze identity roles, then acquire only non-test author ZIPs without decoding."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'datasets/arithmetic_metadata'
OUT = ROOT / 'datasets/arithmetic'
REPO = 'cityuCompuNeuroLab/MentalWorkload_cognitiveEEG'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def make_plan(tree):
    entries = [dict(r) for r in tree['tree'] if r['path'].endswith('.zip')]
    if tree.get('truncated') or len(entries) != 19:
        raise ValueError('Expected complete observed 19-archive release, not paper count20')
    entries.sort(key=lambda r: hashlib.sha256(('20260920:' + r['path']).encode()).hexdigest())
    for index, entry in enumerate(entries):
        if Path(entry['path']).name != entry['path'] or entry['size'] > 10_000_000:
            raise ValueError('Unexpected archive path/size')
        entry['role'] = 'train_fit' if index < 7 else 'train_calibration' if index < 11 else 'dev' if index < 15 else 'test_sealed'
    planned = sum(r['size'] for r in entries if r['role'] != 'test_sealed')
    if planned > 100_000_000:
        raise ValueError('Acquisition reservation exceeded')
    return dict(tree_sha=tree['sha'], entries=entries, planned_bytes=planned,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seed=20260920, paper_subjects=20, release_archives=19,
                scope='Acquisition only; no measurement member decoding',
                session_roles={'T1': 'practice excluded', 'T2-T3': 'training enrollment support',
                               'T4-T6': 'fit/calibration/dev probes according to identity role'},
                modality='Cal only for later separately guarded feature extraction',
                limitations='One visit, ordered difficulty blocks; not cross-day validation. Ordinal levels are not equal physical difficulty increments.',
                source_sha256=sha(Path(__file__)), tree_sha256=sha(META / 'tree.json'))


def prepare():
    plan = make_plan(json.loads((META / 'tree.json').read_text()))
    OUT.mkdir(exist_ok=False)
    (OUT / 'raw').mkdir()
    write(OUT / 'plan.json', plan)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'planned_bytes': plan['planned_bytes'], 'roles':
        {role: [r['path'] for r in plan['entries'] if r['role'] == role]
         for role in ['train_fit', 'train_calibration', 'dev', 'test_sealed']}}))


def acquire():
    plan = json.loads((OUT / 'plan.json').read_text())
    if sha(Path(__file__)) != plan['source_sha256'] or sha(META / 'tree.json') != plan['tree_sha256']:
        raise ValueError('Prepared source/metadata changed')
    if (OUT / 'complete.json').exists():
        raise FileExistsError('Acquisition already completed')
    receipts = []
    for entry in plan['entries']:
        if entry['role'] == 'test_sealed':
            continue
        dest = OUT / 'raw' / entry['path']
        url = 'https://raw.githubusercontent.com/' + REPO + '/' + plan['tree_sha'] + '/' + entry['path']
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read(entry['size'] + 1)
            blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            if len(body) != entry['size'] or blob != entry['sha']:
                raise ValueError('Publisher size/hash mismatch')
            with dest.open('xb') as stream:
                stream.write(body)
        body = dest.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if len(body) != entry['size'] or blob != entry['sha']:
            raise ValueError('Local archive integrity mismatch')
        with zipfile.ZipFile(dest) as archive:
            members = [dict(name=r.filename, bytes=r.file_size, compressed=r.compress_size,
                            crc=r.CRC) for r in archive.infolist()]
        receipts.append(dict(path=entry['path'], role=entry['role'], url=url,
                             sha256=sha(dest), bytes=len(body), members=members))
        print(entry['path'], entry['role'], len(body), flush=True)
    if any((OUT / 'raw' / r['path']).exists() for r in plan['entries'] if r['role'] == 'test_sealed'):
        raise ValueError('Sealed archive unexpectedly present')
    write(OUT / 'complete.json', dict(plan_sha256=sha(OUT / 'plan.json'), archives=receipts,
                                    raw_bytes=sum(r['bytes'] for r in receipts),
                                    measurements_decoded=False, test_downloaded=False))


if __name__ == '__main__':
    {'prepare': prepare, 'acquire': acquire}[sys.argv[1]]()
