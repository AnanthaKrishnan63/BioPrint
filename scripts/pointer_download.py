"""Fetch only preassigned Balabit training/development sessions; never test content."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/benchmarks/balabit'
DEST.mkdir(parents=True, exist_ok=True)
manifest_path = DEST / 'split_manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    if not (DEST / 'tree.json').exists():
        subprocess.run(['curl', '-fLsS', '--connect-timeout', '30', '--max-time', '180', 'https://api.github.com/repos/balabit/Mouse-Dynamics-Challenge/git/trees/master?recursive=1', '-o', str(DEST / 'tree.json')], check=True)
    tree = json.loads((DEST / 'tree.json').read_text())
    entries = [e for e in tree['tree'] if e['path'].startswith('training_files/') and e['type'] == 'blob']
    users = sorted({e['path'].split('/')[1] for e in entries})
    for user in users:
        files = sorted([e for e in entries if e['path'].split('/')[1] == user], key=lambda e: hashlib.sha256(('bioprint-pointer-v1:' + e['path']).encode()).hexdigest())
        for i, entry in enumerate(files):
            entry['split'] = 'test_sealed' if i == 0 else 'dev' if i == 1 else 'train'
            entry['train_role'] = 'calibration' if i == 2 else 'fit' if i > 2 else None
    manifest = {'protocol': 'whole-session hash split before reading contents; chronology unavailable', 'source_commit': tree['sha'], 'entries': entries}
    manifest_path.write_text(json.dumps(manifest, indent=2))

def fetch(entry):
    if entry['split'] == 'test_sealed':
        return
    path = DEST / entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == entry['size']:
        return
    url = f"https://raw.githubusercontent.com/balabit/Mouse-Dynamics-Challenge/{manifest['source_commit']}/{entry['path']}"
    subprocess.run(['curl', '-fLsS', '--connect-timeout', '30', '--max-time', '180', '--retry', '3', url, '-o', str(path)], check=True)
    if path.stat().st_size != entry['size']:
        raise ValueError('Downloaded size mismatch')

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, manifest['entries']))
print(json.dumps({'downloaded_bytes': sum(e['size'] for e in manifest['entries'] if e['split'] != 'test_sealed'), 'test_content_downloaded': False}))
