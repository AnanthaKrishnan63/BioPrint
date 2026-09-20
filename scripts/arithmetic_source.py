"""Metadata-only discovery for the author-released arithmetic task dataset."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/arithmetic_metadata'
REPO = 'cityuCompuNeuroLab/MentalWorkload_cognitiveEEG'


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=25) as response:
        body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError('Metadata exceeds 2 MB')
    with path.open('xb') as stream:
        stream.write(body)
    return body


def main():
    OUT.mkdir(exist_ok=False)
    tree = json.loads(fetch('https://api.github.com/repos/' + REPO +
                           '/git/trees/master?recursive=1', OUT / 'tree.json'))
    if tree.get('truncated'):
        raise ValueError('Incomplete tree')
    fetch('https://raw.githubusercontent.com/' + REPO + '/' + tree['sha'] +
          '/README.md', OUT / 'README.md')
    archives = [r for r in tree['tree'] if r['path'].endswith('.zip')]
    print(json.dumps({'tree_sha': tree['sha'], 'archives': archives,
                      'bytes': sum(r['size'] for r in archives)}))
    receipts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in OUT.iterdir() if p.is_file()}
    (OUT / 'checksums.json').write_text(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
