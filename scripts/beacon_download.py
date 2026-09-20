"""Fetch a bounded, preregistered BEACON subset; test identities never fetched.

The subset is domain-shifted gameplay, not login behavior. Mouse files are raw
16 MiB prefixes and must be clipped to the same time span as paired keyboard data
before feature computation. No screen, packet, or user configuration captures.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import urllib.request
import concurrent.futures

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/beacon'
REVISION = 'da1306428ef626108a914abff42ead19b3a46f62'
BASE = f'https://huggingface.co/datasets/beacon-gui/BEACON-Dataset/resolve/{REVISION}/'
PREFIX_BYTES = 16 * 1024 * 1024


def plan():
    rows = list(csv.DictReader((DATA / 'public_release_manifest.csv').open()))
    sessions = {}
    for row in rows:
        if row['release_path'] and row['modality'] in {'keyboard_csv', 'mouse_csv', 'hardware'}:
            sessions.setdefault(row['participant_id'], {}).setdefault(row['session_id'], {})[row['modality']] = row
    eligible = sorted(p for p, ss in sessions.items() if sum('keyboard_csv' in fs and 'mouse_csv' in fs for fs in ss.values()) >= 2)
    # Manifest-only selection, fixed before seeing feature values.
    train_end, dev_end = int(len(eligible) * .6), int(len(eligible) * .85)
    membership = {p: 'train' if i < train_end else 'dev' if i < dev_end else 'test' for i, p in enumerate(eligible)}
    files = []
    for p in eligible:
        selected = sorted(s for s, fs in sessions[p].items() if 'keyboard_csv' in fs and 'mouse_csv' in fs)[:2]
        for index, s in enumerate(selected):
            for modality in ['keyboard_csv', 'mouse_csv', 'hardware']:
                if modality not in sessions[p][s]:
                    continue
                row = sessions[p][s][modality]
                limit = PREFIX_BYTES if modality == 'mouse_csv' else int(row['size_bytes'])
                files.append({**row, 'split': membership[p], 'role': 'enrollment' if index == 0 else 'probe',
                              'download_bytes': min(limit, int(row['size_bytes'])),
                              'partial': limit < int(row['size_bytes'])})
    result = {'dataset': 'BEACON paired gameplay prefix subset', 'version': 2, 'revision': REVISION,
              'revision_reason': 'Training P002/S001 first key starts186s, beyond2MiB mouse prefix161.9s; uniform16MiB prefix chosen before dev inspection',
              'protocol': 'lexicographically sorted identities with >=2 paired sessions: floor(60%) train, up to floor(85%) dev, remaining test; first two sessions enrollment/probe',
              'test_policy': 'metadata only, files never requested',
              'excluded_single_session_subjects': sorted(set(sessions) - set(eligible)),
              'subjects': membership, 'files': files,
              'planned_download_bytes': sum(r['download_bytes'] for r in files if r['split'] != 'test')}
    path = DATA / 'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != result:
        previous = json.loads(path.read_text())
        if previous['version'] != 1 or previous['subjects'] != result['subjects']:
            raise ValueError('Refusing changed existing split manifest')
        (DATA / 'split_manifest.v1.json').write_text(path.read_text())
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def fetch(row):
    assert row['split'] != 'test'
    path = DATA / row['split'] / row['release_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = DATA / 'source_tree_metadata.json'
    if not metadata_file.exists():
        raise ValueError('Pinned source tree metadata required for integrity validation')
    source = next(m for m in json.loads(metadata_file.read_text()) if m['path'] == row['release_path'])
    source_size = int(source['size'])
    limit = min(PREFIX_BYTES, source_size) if row['modality'] == 'mouse_csv' else source_size
    partial = limit < source_size
    if path.exists() and path.stat().st_size == limit:
        payload = path.read_bytes()
    else:
        headers = {'Range': f'bytes=0-{limit - 1}'} if partial else {}
        request = urllib.request.Request(BASE + row['release_path'], headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if partial and response.status != 206:
                raise ValueError('Server ignored partial request; refusing full download')
            # Published metadata has stale sizes/hashes for some corrected CSVs.
            # Bound full non-mouse files to 8MiB; retain discrepancy in receipt.
            cap = limit
            payload = response.read(cap + 1)
            content_range = response.headers.get('Content-Range', '')
        valid_short = False
        if partial and len(payload) < limit:
            match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', content_range)
            valid_short = bool(match and int(match[1]) + 1 == len(payload) == int(match[2]))
        if len(payload) != limit:
            raise ValueError(f'Unexpected response size for {path}')
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    git_digest = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    verified_source = None if partial else (digest == source['lfs']['oid'] if 'lfs' in source else git_digest == source['oid'])
    if verified_source is False:
        raise ValueError(f'Pinned source checksum mismatch: {path}')
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(payload),
            'sha256': digest, 'partial': partial, 'split': row['split'],
            'pinned_source_integrity_verified': verified_source,
            'published_manifest_sha256_match': None if partial else digest == row['sha256'],
            'published_size': int(row['size_bytes']), 'revision': REVISION}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    manifest = plan()
    print(json.dumps({'subjects': manifest['subjects'], 'planned_bytes': manifest['planned_download_bytes']}), flush=True)
    if args.download:
        metadata = DATA / 'source_tree_metadata.json'
        if not metadata.exists():
            url = f'https://huggingface.co/api/datasets/beacon-gui/BEACON-Dataset/tree/{REVISION}?recursive=true&limit=1000'
            with urllib.request.urlopen(url, timeout=60) as response:
                metadata.write_bytes(response.read(2000000))
        allowed = [r for r in manifest['files'] if r['split'] != 'test']
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            downloaded = list(pool.map(fetch, allowed))
        (DATA / 'download_receipts.json').write_text(json.dumps(downloaded, indent=2) + '\n')
        print(json.dumps({'downloaded_files': len(downloaded), 'bytes': sum(r['bytes'] for r in downloaded)}))
