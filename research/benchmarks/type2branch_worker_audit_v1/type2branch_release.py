"""Hash-verified fixed-path research release; no TensorFlow or dataset loading."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'research/benchmarks/type2branch_release_v1'
FILES=frozenset(['encoder.index','encoder.data-00000-of-00001','profiles.npz','dev_features.npz','dev_report.json'])
SOURCES=frozenset(['scripts/type2branch_worker.py','scripts/type2branch_wire.py',
    'scripts/type2branch_scoring.py','scripts/type2branch_release.py',
    'research/benchmarks/references/type2branch/model.py',
    'research/benchmarks/references/type2branch/conf.small.1Kusers.py',
    '.research-type2branch-deps/tf_keras/src/backend.py'])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_release(expected_sha, directory=RELEASE):
    if not isinstance(expected_sha,str) or re.fullmatch('[0-9a-f]{64}',expected_sha) is None:
        raise ValueError('Pinned release SHA256 required')
    raw=(directory/'manifest.json').read_bytes()
    if digest(raw)!=expected_sha:raise ValueError('Release manifest mismatch')
    manifest=json.loads(raw)
    if manifest.get('version')!=1 or manifest.get('split')!='dev':raise ValueError('Wrong release version/split')
    if set(manifest['files'])!=FILES or set(manifest['sources'])!=SOURCES:
        raise ValueError('Release file/source inventory mismatch')
    for name,expected in manifest['files'].items():
        if digest((directory/name).read_bytes())!=expected:raise ValueError('Release artifact changed')
    for name,expected in manifest['sources'].items():
        if digest((ROOT/name).read_bytes())!=expected:raise ValueError('Release inference source changed')
    return manifest
