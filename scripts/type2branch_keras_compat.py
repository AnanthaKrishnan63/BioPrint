"""Record a one-literal Python3.12 compatibility patch to local legacy Keras."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / '.research-type2branch-deps/tf_keras/src/backend.py'
    out = ROOT / 'research/benchmarks/type2branch_runtime_v1/keras_compat.json'
    if out.exists():
        raise FileExistsError('Compatibility patch already recorded')
    before = path.read_bytes()
    old = b'_SEED_GENERATOR.generator.randint(1, 1e9)'
    new = b'_SEED_GENERATOR.generator.randint(1, 1000000000)'
    if before.count(old) != 1:
        raise ValueError('Expected exactly one reviewed seed bound')
    after = before.replace(old, new)
    record = {'path': str(path.relative_to(ROOT)),
              'before_sha256': hashlib.sha256(before).hexdigest(),
              'after_sha256': hashlib.sha256(after).hexdigest(),
              'change': 'Integral float upper bound 1e9 to integer1000000000',
              'reason': 'Python3.12 random.randrange no longer accepts integral floats',
              'limitation': 'Explicit dependency adaptation; not unmodified author environment'}
    (out.parent / 'keras_backend_original.py').write_bytes(before)
    path.write_bytes(after)
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
