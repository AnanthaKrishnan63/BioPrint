"""Capture bounded official KMT metadata; never download observation files."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'landing': 'https://data.mendeley.com/datasets/fnf8b85kr6/1',
    'public_version': 'https://data.mendeley.com/public-api/datasets/fnf8b85kr6/versions/1',
    'public_files': 'https://data.mendeley.com/public-api/datasets/fnf8b85kr6/files?folder_id=root&version=1',
}


def main():
    label = sys.argv[1]
    if not label.isalnum():
        raise ValueError('Audit label must be alphanumeric')
    out = ROOT / 'datasets/kmt_metadata' / label
    out.mkdir(exist_ok=False)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    receipts = []
    for name, url in URLS.items():
        row = dict(name=name, url=url, checked_at=datetime.datetime.now(
            datetime.timezone.utc).isoformat(), scope='metadata only')
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                body = response.read(1_000_001)
                if len(body) > 1_000_000:
                    raise ValueError('Metadata exceeds 1 MB bound')
                row.update(status=response.status, bytes=len(body),
                           sha256=hashlib.sha256(body).hexdigest())
            (out / (name + '.response')).write_bytes(body)
        except Exception as exc:
            row['error'] = str(exc)
        receipts.append(row)
        (out / 'receipts.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
