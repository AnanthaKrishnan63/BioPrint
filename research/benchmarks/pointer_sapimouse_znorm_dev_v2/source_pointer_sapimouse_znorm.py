"""Frozen FCN cosine / enrollment Z-norm TRAIN experiment; no DEV loader.

Z-norm is a verification-backend transfer hypothesis, not mouse SOTA. Impostor
statistics use only representation-TRAIN background recordings and the claimed
account's enrollment. Probe observations never estimate normalization.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pointer_sapimouse_benchmark import FCN, load, embed, aggregate, DATA
from pointer_sapimouse_optimize import latent_scores
from pointer_benchmark import stats, threshold_at_far

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/pointer_sapimouse_znorm_v1'
CHECKPOINT = ROOT / 'research/benchmarks/pointer_sapimouse/fcn_training_best.pt'
MANIFEST = DATA / 'split_manifest.json'
SOURCE_URL = 'https://www.isca-archive.org/interspeech_2018/shi18b_interspeech.pdf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_entries(manifest):
    result = {}
    for label, role, minutes in [('background', 'representation', 3),
                                 ('enrollment', 'calibration', 3), ('calibration', 'calibration', 1)]:
        entries = [e for e in manifest['files'] if e['split'] == 'train'
                   and e['role'] == role and e['path'].endswith(f'_{minutes}min.csv')]
        for entry in entries:
            if not (DATA / entry['path']).resolve().is_relative_to(DATA.resolve()):
                raise ValueError('Source path escapes dataset')
        identities = {e['path'].split('/')[1] for e in entries}
        if len(identities) != (60 if label == 'background' else 12):
            raise ValueError('Fixed TRAIN identity population changed')
        result[label] = entries
    background = {e['path'].split('/')[1] for e in result['background']}
    enrolled = {e['path'].split('/')[1] for e in result['enrollment']}
    calibrated = {e['path'].split('/')[1] for e in result['calibration']}
    if background & enrolled or enrolled != calibrated:
        raise ValueError('Background identities must be disjoint from calibration accounts')
    return result


def fit_znorm(background_scores):
    scores = np.asarray(background_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Finite per-account TRAIN background scores required')
    mean, spread = scores.mean(axis=1), scores.std(axis=1, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError('No usable background spread; no verdict')
    return mean, spread


def normalize(scores, mean, spread):
    scores, mean, spread = [np.asarray(v, dtype=np.float64) for v in (scores, mean, spread)]
    if (scores.ndim != 2 or mean.shape != (scores.shape[0],) or spread.shape != mean.shape
            or not all(np.isfinite(v).all() for v in (scores, mean, spread)) or np.any(spread <= 0)):
        raise ValueError('Frozen finite per-account normalization required')
    result = (scores - mean[:, None]) / spread[:, None]
    if not np.isfinite(result).all(): raise ValueError('Nonfinite normalization')
    return result


def prepare():
    entries = training_entries(json.loads(MANIFEST.read_text()))
    source_names = ['pointer_sapimouse_znorm.py', 'pointer_sapimouse_benchmark.py',
                    'pointer_sapimouse_optimize.py', 'pointer_benchmark.py']
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256': digest(CHECKPOINT), 'manifest_sha256': digest(MANIFEST),
            'source_sha256': {n: digest(ROOT / 'scripts' / n) for n in source_names},
            'engine_scorer_sha256': digest(ROOT / 'code/bioprint/engine/scorer.py'),
            'entries': entries, 'methods': ['cosine@5', 'znorm_cosine@5'],
            'blocks': 5, 'events_per_decision': 641, 'cpu_threads': 2,
            'background': 'All 60 representation-TRAIN identities, 3min files only; no calibration/DEV/test identities',
            'normalization': 'Per enrolled account mean/std(ddof0) of its cosine scores against nonoverlapping 5-block TRAIN-background groups; zero spread fails, no imputation',
            'selection': 'Minimum calibration FRR at TRAIN-calibrated1% FAR; tie calibration EER, then listed method order',
            'targets': [.001, .01, .05], 'source': SOURCE_URL,
            'scope': 'Backend transfer hypothesis on frozen source FCN; no modern mouse SOTA claim',
            'prior_exposure': 'Original SapiMouse DEV results already reported. This run reads TRAIN only; any follow-up DEV is not pristine.',
            'no_new_encoder_fit': True, 'no_dev_or_test_access': True}
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    for name in source_names:
        (OUT / ('source_' + name)).write_bytes((ROOT / 'scripts' / name).read_bytes())
    print('Prepared TRAIN-only experiment; no recording values read')


def run():
    plan = json.loads((OUT / 'plan.json').read_text())
    if digest(CHECKPOINT) != plan['checkpoint_sha256'] or digest(MANIFEST) != plan['manifest_sha256']:
        raise ValueError('Frozen checkpoint/role manifest changed')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != expected: raise ValueError('Frozen source changed')
    if digest(ROOT / 'code/bioprint/engine/scorer.py') != plan['engine_scorer_sha256']:
        raise ValueError('Frozen source dependency changed')
    entries = training_entries(json.loads(MANIFEST.read_text()))
    if entries != plan['entries']: raise ValueError('Prepared TRAIN records changed')
    (OUT / 'run_started.json').open('x').close()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = FCN(saved['classes']).eval(); model.load_state_dict(saved['state_dict'])
    arrays = {}
    for kind, records in entries.items():
        blocks, _, labels, sessions = load(records)
        arrays[kind] = (embed(model, blocks), labels, sessions)
    enrollment, labels, _ = arrays['enrollment']
    probe, probe_labels, probe_sessions = arrays['calibration']
    background, background_labels, background_sessions = arrays['background']
    raw, accounts = latent_scores(enrollment, labels, probe, 'cosine')
    raw, actual, sessions = aggregate(raw, probe_labels, probe_sessions, 5)
    background_raw, background_accounts = latent_scores(enrollment, labels, background, 'cosine')
    if not np.array_equal(accounts, background_accounts): raise ValueError('Account order changed')
    background_scores, _, _ = aggregate(background_raw, background_labels, background_sessions, 5)
    mean, spread = fit_znorm(background_scores)
    methods = {'cosine@5': raw, 'znorm_cosine@5': normalize(raw, mean, spread)}
    truth = (accounts[:, None] == actual[None, :]).ravel().astype(int)
    reports, thresholds = {}, {}
    for method, scores in methods.items():
        thresholds[method] = {str(t): float(threshold_at_far(truth, scores.ravel(), t)) for t in plan['targets']}
        reports[method] = {t: stats(truth, scores.ravel(), value) for t, value in thresholds[method].items()}
    selected = min(plan['methods'], key=lambda n: (reports[n]['0.01']['frr'], reports[n]['0.01']['eer_descriptive']))
    with (OUT / 'train_scores.npz').open('xb') as f:
        np.savez_compressed(f, cosine=raw, znorm=methods['znorm_cosine@5'], accounts=accounts,
                            actual=actual, sessions=sessions, background_scores=background_scores,
                            mean=mean, spread=spread)
    report = {'status': 'complete_train_only', 'plan': plan, 'selected': selected, 'reports': reports,
              'thresholds': thresholds, 'calibration_decisions': len(actual),
              'background_decisions': background_scores.shape[1],
              'scores_sha256': digest(OUT / 'train_scores.npz'), 'dev_or_test_read': False}
    (OUT / 'train_complete.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'selected': selected, 'calibration': {k: v['0.01'] for k, v in reports.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args(); prepare() if args.action == 'prepare' else run()
