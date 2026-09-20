"""Document frozen-parser DEV failures without relaxing rules or scoring subsets."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import loadmat
from arithmetic_features import parse_block
from arithmetic_dev import dev_records
from arithmetic_train import sha, write

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/arithmetic'
DEV = ROOT / 'research/benchmarks/arithmetic_dev_v1'
OUT = ROOT / 'research/benchmarks/arithmetic_dev_failure_v1'


def check_block(data):
    try:
        parse_block(data)
        return {'status': 'valid'}
    except ValueError as exc:
        return {'status': 'invalid', 'error': str(exc),
                'shapes': {key: list(np.shape(data[key])) for key in ['imageT', 'keyT', 'CorrectAns']}}


def main():
    plan = json.loads((DEV / 'plan.json').read_text())
    for name, expected in plan['inputs'].items():
        if sha(ROOT / name) != expected: raise ValueError('Frozen input changed')
    for name, expected in plan['sources'].items():
        if sha(ROOT / 'scripts' / name) != expected: raise ValueError('Frozen source changed')
    acquisition = json.loads((DATA / 'plan.json').read_text())
    if plan['records'] != dev_records(acquisition): raise ValueError('DEV scope changed')
    receipt = json.loads((DATA / 'complete.json').read_text())
    hashes = {r['path']: r['sha256'] for r in receipt['archives']}
    OUT.mkdir(exist_ok=False)
    write(OUT / 'plan.json', dict(original_plan_sha256=sha(DEV / 'plan.json'),
          source_sha256=sha(Path(__file__)), scope='Evaluation-only original parser diagnostics; no scoring, no rules changed'))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for row in plan['records']:
        path = DATA / 'raw' / row['path']
        if sha(path) != hashes[row['path']]: raise ValueError('Archive changed')
        subject = path.stem
        with zipfile.ZipFile(path) as archive:
            for trial in range(2, 7):
                for level in 'lmh':
                    member = f'Cal_{subject}_L{level}T{trial}.mat'
                    if archive.getinfo(member).file_size > 2_000_000: raise ValueError('Member bound')
                    data = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)['Data']
                    rows.append(dict(subject=subject, round=trial, level=level, member=member,
                                     role='training_enrollment_support' if trial < 4 else 'dev_probe',
                                     **check_block(data)))
    invalid = [row for row in rows if row['status'] != 'valid']
    result = dict(status='infeasible' if invalid else 'diagnostic_disagrees_with_prior_failure',
                  selected=plan['selected'], checked_blocks=len(rows), invalid_blocks=invalid,
                  metrics={}, test_accessed=False, parser_changed=False,
                  subset_scored=False, original_execution_exit_code=1)
    write(OUT / 'report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
