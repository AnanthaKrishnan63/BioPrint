"""Evaluation-only arithmetic DEV, using frozen TRAIN metrics and thresholds."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat
from arithmetic_features import profile
from arithmetic_train import FEATURES, METHODS, score, stats, labels, sha, write

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/arithmetic'
TRAIN = ROOT / 'research/benchmarks/arithmetic_train_v1'
OUT = ROOT / 'research/benchmarks/arithmetic_dev_v1'


def dev_records(acquisition):
    rows = sorted([r for r in acquisition['entries'] if r['role'] == 'dev'], key=lambda r: r['path'])
    if len(rows) != 4 or len({r['path'] for r in rows}) != 4:
        raise ValueError('Expected all four DEV accounts')
    if any(Path(r['path']).name != r['path'] for r in rows):
        raise ValueError('Archive path escape')
    return rows


def prepare():
    report = json.loads((TRAIN / 'train_complete.json').read_text())
    acquisition = json.loads((DATA / 'plan.json').read_text())
    sources = ['arithmetic_dev.py', 'arithmetic_train.py', 'arithmetic_features.py']
    OUT.mkdir(exist_ok=False)
    write(OUT / 'plan.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in
        [TRAIN / 'train_complete.json', TRAIN / 'selection.json', DATA / 'plan.json', DATA / 'complete.json']},
        sources={name: sha(ROOT / 'scripts' / name) for name in sources},
        records=dev_records(acquisition), selected=report['selected'],
        thresholds=report['thresholds'], evaluation_only=True,
        enrollment='T2/T3 personal support only; existing TRAIN global parameters',
        probes='T4-T6, all three difficulty levels;12genuine/36impostor claims expected',
        failure_policy='Whole comparison infeasible if any required block fails fixed parser; no DEV-based exception',
        test_accessed=False))
    for name in sources:
        (OUT / ('source_' + name)).write_bytes((ROOT / 'scripts' / name).read_bytes())


def run():
    if (OUT / 'dev_complete.json').exists():
        raise FileExistsError('DEV comparison already completed')
    plan = json.loads((OUT / 'plan.json').read_text())
    for name, expected in plan['inputs'].items():
        if sha(ROOT / name) != expected: raise ValueError('Input changed')
    for name, expected in plan['sources'].items():
        if sha(ROOT / 'scripts' / name) != expected: raise ValueError('Source changed')
    train = json.loads((TRAIN / 'train_complete.json').read_text())
    acquisition = json.loads((DATA / 'plan.json').read_text())
    receipt = json.loads((DATA / 'complete.json').read_text())
    if plan['records'] != dev_records(acquisition) or plan['selected'] != train['selected'] or plan['thresholds'] != train['thresholds']:
        raise ValueError('Plan differs from frozen roles/selection/thresholds')
    hashes = {r['path']: r['sha256'] for r in receipt['archives']}
    accounts, features = [], {key: [] for key in FEATURES}
    for row in plan['records']:
        path = DATA / 'raw' / row['path']
        if sha(path) != hashes[row['path']]: raise ValueError('Archive changed')
        account = path.stem
        rounds = []
        with zipfile.ZipFile(path) as archive:
            for trial in range(2, 7):
                blocks = {}
                for level in 'lmh':
                    member = f'Cal_{account}_L{level}T{trial}.mat'
                    if archive.getinfo(member).file_size > 2_000_000: raise ValueError('Member size bound')
                    blocks[level] = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)['Data']
                rounds.append(profile(blocks))
        accounts.append(account)
        for key in FEATURES: features[key].append([r[key] for r in rounds])
    arrays = {'accounts': np.asarray(accounts), 'actual': np.repeat(accounts, 3),
              'probe_rounds': np.tile([4, 5, 6], len(accounts))}
    results = {}
    for feature in FEATURES:
        x = np.asarray(features[feature], dtype=float)
        arrays[feature + '_templates'] = x[:, :2].mean(axis=1)
        arrays[feature + '_queries'] = x[:, 2:].reshape(-1, x.shape[-1])
    for feature, metric in METHODS:
        name = feature + '__' + metric
        matrix = score(metric, train['parameters'][feature], arrays[feature + '_templates'], arrays[feature + '_queries'])
        arrays[name] = matrix
        results[name] = {target: stats(labels(accounts, 3), matrix.ravel(), cutoff)
                         for target, cutoff in train['thresholds'][name].items()}
    np.savez_compressed(OUT / 'scores.npz', **arrays)
    write(OUT / 'dev_complete.json', dict(status='complete_evaluation_only', plan=plan,
          selected=train['selected'], selection_status=train['selection_status'],
          methods=results, accounts=accounts, probe_decisions=12,
          scores_sha256=sha(OUT / 'scores.npz'), test_accessed=False,
          dev_tuning=False, all_feature_fusion=False))
    print(json.dumps({'selected': train['selected'], 'methods_at_1pct':
                      {key: value['0.01'] for key, value in results.items()}}))


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
