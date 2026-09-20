"""Acquire the official pure-TensorFlow distance source, without Addons binaries."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/references/type2branch_addons'


def main():
    OUT.mkdir(exist_ok=False)
    receipts = []
    def fetch(url, name):
        with urllib.request.urlopen(url, timeout=25) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError('Source metadata bound exceeded')
        (OUT / name).write_bytes(body)
        receipts.append({'url': url, 'file': name, 'sha256': hashlib.sha256(body).hexdigest()})
        return body
    commit = json.loads(fetch('https://api.github.com/repos/tensorflow/addons/commits/v0.23.0',
                              'commit.json'))['sha']
    prefix = 'https://raw.githubusercontent.com/tensorflow/addons/' + commit + '/'
    fetch(prefix + 'tensorflow_addons/losses/metric_learning.py', 'metric_learning.py')
    fetch(prefix + 'LICENSE', 'LICENSE')
    (OUT / 'receipts.json').write_text(json.dumps({'commit': commit, 'files': receipts}, indent=2))
    print(json.dumps({'commit': commit, 'files': len(receipts)}))


if __name__ == '__main__':
    main()
