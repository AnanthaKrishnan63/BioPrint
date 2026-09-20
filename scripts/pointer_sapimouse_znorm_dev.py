"""Frozen comparison-only DEV validation of the unselected Z-norm candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pointer_sapimouse_znorm import (ROOT, CHECKPOINT, MANIFEST, digest, training_entries,
                                    fit_znorm, normalize)
from pointer_sapimouse_benchmark import FCN, load, embed, aggregate
from pointer_benchmark import stats

TRAIN = ROOT / 'research/benchmarks/pointer_sapimouse_znorm_v1'
OLD = ROOT / 'research/benchmarks/pointer_sapimouse_optimized'
OUT = ROOT / 'research/benchmarks/pointer_sapimouse_znorm_dev_v2'
SOURCES = ['pointer_sapimouse_znorm_dev.py', 'pointer_sapimouse_znorm.py',
           'pointer_sapimouse_benchmark.py', 'pointer_sapimouse_optimize.py', 'pointer_benchmark.py']


def validate_records(plan, manifest, train):
    background = training_entries(manifest)['background']
    dev = [e for e in manifest['files'] if e['split'] == 'dev' and e['role'] == 'dev_probe']
    subjects = sorted({e['path'].split('/')[1] for e in manifest['files']
                       if e['split'] == 'train' and e['role'] == 'dev_support_enrollment'})
    if (plan['background_entries'] != background or plan['dev_entries'] != dev
            or plan['subjects'] != subjects or len(subjects) != 24
            or plan['thresholds'] != train['thresholds'] or plan['selected'] != 'cosine@5'):
        raise ValueError('Prepared record roles, accounts, or frozen thresholds changed')


def prepare():
    train = json.loads((TRAIN / 'train_complete.json').read_text())
    if train['selected'] != 'cosine@5' or train['status'] != 'complete_train_only':
        raise ValueError('Preserve original TRAIN outcome')
    if digest(CHECKPOINT) != train['plan']['checkpoint_sha256'] or digest(MANIFEST) != train['plan']['manifest_sha256']:
        raise ValueError('Training sources changed')
    manifest = json.loads(MANIFEST.read_text())
    subjects = sorted({e['path'].split('/')[1] for e in manifest['files']
                       if e['split'] == 'train' and e['role'] == 'dev_support_enrollment'})
    entries = [e for e in manifest['files'] if e['split'] == 'dev' and e['role'] == 'dev_probe']
    if len(subjects) != 24 or {e['path'].split('/')[1] for e in entries} != set(subjects):
        raise ValueError('All24 original DEV accounts required')
    old = json.loads((OLD / 'frozen_training_selection.json').read_text())
    if old['selected'] != 'cosine@5' or old['thresholds']['cosine@5'] != train['thresholds']['cosine@5']:
        raise ValueError('Original selected scorer changed')
    inputs = [CHECKPOINT, MANIFEST, TRAIN / 'train_complete.json', OLD / 'frozen_training_selection.json',
              *[OLD / f'cosine_{u}.npz' for u in subjects]]
    plan = {'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            'source_sha256': {n: digest(ROOT / 'scripts' / n) for n in SOURCES},
            'subjects': subjects, 'dev_entries': entries,
            'background_entries': training_entries(manifest)['background'],
            'thresholds': train['thresholds'], 'selected': 'cosine@5',
            'evaluation_only': True, 'no_dev_selection_or_tuning': True,
            'prior_dev_exposure': 'Original SapiMouse DEV already evaluated; this is a frozen comparison extension.',
            'normalization': 'Only existing TRAIN-support centroids and representation-TRAIN background scores; no DEV-probe normalization fit',
            'blocks': 5, 'api_tolerance': {'raw_atol': 1e-5, 'znorm_atol': 1e-4, 'rtol': 1e-5},
            'test_accessed': False}
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    for n in SOURCES: (OUT / ('source_' + n)).write_bytes((ROOT / 'scripts' / n).read_bytes())
    print('Prepared frozen comparison; no recording values read')


def run():
    plan = json.loads((OUT / 'plan.json').read_text())
    for path, sha in plan['input_sha256'].items():
        if digest(ROOT / path) != sha: raise ValueError('Frozen input changed')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != sha: raise ValueError('Frozen source changed')
    validate_records(plan, json.loads(MANIFEST.read_text()), json.loads((TRAIN / 'train_complete.json').read_text()))
    (OUT / 'run_started.json').open('x').close()
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.use_deterministic_algorithms(True)
    saved = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    model = FCN(saved['classes']).eval(); model.load_state_dict(saved['state_dict'])
    centers = []
    for user in plan['subjects']:
        with np.load(OLD / f'cosine_{user}.npz', allow_pickle=False) as a: centers.append(a['center'].copy())
    centers = np.asarray(centers)
    if centers.shape != (24, 128) or not np.isfinite(centers).all(): raise ValueError('Invalid original centroids')
    def score(blocks):
        embeddings = embed(model, blocks)
        unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        return np.asarray([unit @ c for c in centers])
    background, _, blabels, bgroups = load(plan['background_entries'])
    bscore, _, _ = aggregate(score(background), blabels, bgroups, 5)
    mean, spread = fit_znorm(bscore)
    blocks, _, labels, groups = load(plan['dev_entries'])
    raw, actual, sessions = aggregate(score(blocks), labels, groups, 5)
    normalized = normalize(raw, mean, spread)
    accounts = np.asarray(plan['subjects'])
    truth = (accounts[:, None] == actual[None, :]).ravel().astype(int)
    reports = {}
    for name, scores in [('cosine@5', raw), ('znorm_cosine@5', normalized)]:
        reports[name] = {t: stats(truth, scores.ravel(), threshold)
                         for t, threshold in plan['thresholds'][name].items()}
    with (OUT / 'scores.npz').open('xb') as f:
        np.savez_compressed(f, cosine=raw, znorm=normalized, accounts=accounts, actual=actual, sessions=sessions)
    with (OUT / 'normalization.npz').open('xb') as f:
        np.savez_compressed(f, accounts=accounts, mean=mean, spread=spread)
    report = {'status': 'complete_evaluation_only', 'plan': plan, 'selected': 'cosine@5',
              'dev': reports, 'dev_blocks': len(blocks), 'dev_decisions': len(actual),
              'scores_sha256': digest(OUT / 'scores.npz'), 'normalization_sha256': digest(OUT / 'normalization.npz'),
              'test_accessed': False, 'candidate_promoted': False}
    (OUT / 'dev_complete.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({m: points['0.01'] for m, points in reports.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run'])
    args = p.parse_args(); prepare() if args.action == 'prepare' else run()
