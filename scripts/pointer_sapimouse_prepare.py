"""Fetch archive, freeze identity split using names only, extract no test contents."""
import hashlib
import json
import pathlib
import subprocess
import zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/benchmarks/sapimouse'
DEST.mkdir(parents=True, exist_ok=True)
archive = DEST / 'source.zip'
if not archive.exists():
    subprocess.run(['curl', '-fLsS', '--connect-timeout', '30', '--max-time', '180', '--max-filesize', '500000000', 'https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip', '-o', str(archive)], check=True)
with zipfile.ZipFile(archive) as source:
    entries = [i for i in source.infolist() if i.filename.endswith('.csv') and '/user' in i.filename]
    users = sorted({i.filename.split('/')[1] for i in entries}, key=lambda u: hashlib.sha256(('sapimouse-v1:' + u).encode()).hexdigest())
    if len(users) != 120:
        raise ValueError('Unexpected subject count')
    if sum(i.file_size for i in entries) > 500_000_000:
        raise ValueError('Expanded archive exceeds per-dataset budget')
    manifest_path = DEST / 'split_manifest.json'
    if not manifest_path.exists():
        assignments = {u: {'split': 'train' if n < 72 else 'dev' if n < 96 else 'test_sealed', 'role': 'representation' if n < 60 else 'calibration' if n < 72 else 'evaluation'} for n, u in enumerate(users)}
        manifest = {'source': 'https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip', 'archive_bytes': archive.stat().st_size, 'expanded_bytes': sum(i.file_size for i in entries), 'protocol': 'user-disjoint hash split before contents: 60 representation train, 12 train calibration, 24 dev, 24 sealed test; each user enrollment3min/probe1min', 'users': assignments, 'files': [{'path': i.filename, 'size': i.file_size, 'crc': i.CRC, **assignments[i.filename.split('/')[1]]} for i in entries]}
        for entry in manifest['files']:
            entry['outer_cohort'] = entry['split']
            if entry['split'] == 'dev':
                if entry['path'].endswith('_3min.csv'):
                    entry['split'] = 'train'
                    entry['role'] = 'dev_support_enrollment'
                else:
                    entry['role'] = 'dev_probe'
        manifest['protocol'] = 'record-level train/dev/test:60representation users;12train calibration;24unseen-encoder dev-cohort users with3min TRAINING-support files and1min DEV-probe files;24sealed-test users'
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['files']:
        if entry['split'] == 'test_sealed':
            continue
        path = DEST / entry['path']
        if not path.resolve().is_relative_to(DEST.resolve()):
            raise ValueError('Unsafe archive path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(source.read(entry['path']))
print(manifest['protocol'])
