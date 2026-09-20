"""Download immutable-checksummed public FPStalker archives without extraction."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'datasets/fpstalker'
BASE = 'https://raw.githubusercontent.com/Spirals-Team/FPStalker/master/'
FILES = {
    'extension1.txt.tar.gz': (73169899, '4791cdbe989e21195862813a3e8ca69dce53e1211a7c4b191eac54d81a712b3f'),
    'extension2.txt.tar.gz': (69956820, '2ea55958c4dde4722014cc0766752c2f254de6159b660341bd9f7c9313de7f42'),
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        path = DEST / name
        if path.exists():
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen(BASE + name, timeout=90) as response:
                raw = response.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Integrity failure: {name}')
        if not path.exists():
            path.write_bytes(raw)
        print(f'{name}: verified {size} bytes')
    for remote, local in [('LICENSE', 'LICENSE'), ('README.md', 'README.upstream.md'),
                          ('extensionDataScheme.sql', 'schema.sql')]:
        path = DEST / local
        if not path.exists():
            with urllib.request.urlopen(BASE + remote, timeout=30) as response:
                raw = response.read(100000)
            path.write_bytes(raw)


if __name__ == '__main__':
    main()
