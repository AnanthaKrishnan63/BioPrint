"""Snapshot bounded public issue and branch metadata; no dataset access."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/type2branch_upstream_audit_v1'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    endpoints = {'issue2': 'issues/2', 'comments2': 'issues/2/comments?per_page=100',
                 'issues_all': 'issues?state=all&per_page=100', 'branches': 'branches?per_page=100'}
    receipts = []
    for name, suffix in endpoints.items():
        url = 'https://api.github.com/repos/lsia/tifs-type2branch/' + suffix
        request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError('Public metadata size bound exceeded')
            link = response.headers.get('Link', '')
        json.loads(body)
        (OUT / (name + '.json')).write_bytes(body)
        receipts.append({'url': url, 'file': name + '.json', 'bytes': len(body),
                         'sha256': hashlib.sha256(body).hexdigest(), 'pagination': link})
    (OUT / 'receipts.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps(receipts))


if __name__ == '__main__':
    main()
