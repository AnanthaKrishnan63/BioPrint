"""Inspect only preregistered TRAIN-fit arithmetic members, never DEV/test."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/arithmetic'
OUT = ROOT / 'research/benchmarks/arithmetic_schema_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_scope(schema, acquisition):
    if schema['permitted_role'] != 'train_fit':
        raise ValueError('Only TRAIN-fit schema reads permitted')
    allowed = {r['path'] for r in acquisition['entries'] if r['role'] == 'train_fit'}
    if len(allowed) != 7 or {r['archive'] for r in schema['entries']} != allowed:
        raise ValueError('Schema cohort differs from frozen fit roles')
    if len(schema['entries']) != 7:
        raise ValueError('Duplicate or missing archive')
    for row in schema['entries']:
        if Path(row['archive']).name != row['archive']:
            raise ValueError('Path escape')
        subject = Path(row['archive']).stem
        expected = {f'Cal_{subject}_L{level}T{trial}.mat'
                    for level in 'lmh' for trial in range(2, 7)}
        if row['subject'] != subject or set(row['members']) != expected or len(row['members']) != 15:
            raise ValueError('Unexpected modality/practice/member scope')


def describe(value, key=''):
    if isinstance(value, dict):
        return {k: describe(v, k) for k, v in value.items()
                if not k.startswith('__') and k.lower() != 'eeg'}
    if isinstance(value, list):
        return {'type': 'list', 'length': len(value),
                'items': [describe(v, key) for v in value]}
    array = np.asarray(value)
    result = {'shape': list(array.shape), 'dtype': str(array.dtype)}
    if array.dtype.kind in 'fiub' and array.size and key in ['keyT', 'imageT', 'CorrectAns']:
        finite = array[np.isfinite(array)]
        result.update(finite=int(finite.size), total=int(array.size),
                      minimum=float(finite.min()) if finite.size else None,
                      maximum=float(finite.max()) if finite.size else None)
    return result


def prepare():
    schema = json.loads((DATA / 'schema_plan_v1.json').read_text())
    acquisition = json.loads((DATA / 'plan.json').read_text())
    validate_scope(schema, acquisition)
    if schema['acquisition_plan_sha256'] != sha(DATA / 'plan.json'):
        raise ValueError('Acquisition plan binding failed')
    OUT.mkdir(exist_ok=False)
    write(OUT / 'plan.json', dict(inputs={name: sha(DATA / name) for name in
          ['schema_plan_v1.json', 'plan.json', 'complete.json']}, source_sha256=sha(Path(__file__))))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())


def run():
    if (OUT / 'report.json').exists():
        raise FileExistsError('Audit already complete')
    plan = json.loads((OUT / 'plan.json').read_text())
    if plan['source_sha256'] != sha(Path(__file__)):
        raise ValueError('Source changed')
    for name, expected in plan['inputs'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Input changed')
    schema = json.loads((DATA / 'schema_plan_v1.json').read_text())
    acquisition = json.loads((DATA / 'plan.json').read_text())
    validate_scope(schema, acquisition)
    result = []
    for row in schema['entries']:
        path = DATA / 'raw' / row['archive']
        if sha(path) != row['sha256']:
            raise ValueError('Archive changed')
        with zipfile.ZipFile(path) as archive:
            for member in row['members']:
                info = archive.getinfo(member)
                if info.file_size > 2_000_000:
                    raise ValueError('Unexpected expanded member size')
                value = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)
                result.append(dict(subject=row['subject'], member=member, schema=describe(value)))
    write(OUT / 'report.json', dict(plan=plan, records=result, members=len(result),
                                   dev_accessed=False, test_accessed=False))
    print(json.dumps({'members': len(result), 'first_schema': result[0]}))


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
