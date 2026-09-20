"""Frozen paired tap/encoder TRAIN-selection runner; no raw or DEV loading.

Prepare pins paths, protocols and executable sources before measurements. Run
only forms personal enrollment galleries; no encoder, scales or threshold fit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hmog_select import embed_all, load_checkpoint
from hmog_training_core import load_author_classes
from hmog_tap_benchmark import build_profiles, score_windows, validate_taps
from hmog_score_fusion import fuse
from hmog_verification import mean_gallery, euclidean_scores
from hmog_tap_selection_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ['180679', '622852']
META = ('subject', 'session', 'role', 'activity_id', 'window_index')
METHODS = ('joint_tap11', 'tap11', 'key', 'imu', 'joint', 'key_imu_tap11')
BRANCHES = ('key', 'imu', 'joint')
FUSIONS = {'joint_tap11': ('joint', 'tap11'), 'key_imu_tap11': ('key', 'imu', 'tap11')}
PATH_ARGS = ('features', 'tap_features', 'tap_report', 'paired_dir', 'selection_dir', 'protocol', 'output')
SOURCES = ('hmog_tap_selection.py', 'hmog_tap_selection_metrics.py', 'hmog_select.py',
           'hmog_training_core.py', 'hmog_verification.py', 'hmog_tap_benchmark.py',
           'hmog_tap_reference.py', 'hmog_score_fusion.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_paths(args):
    return {name: str(getattr(args, name).resolve()) for name in PATH_ARGS}


def validate_metadata(meta):
    n = len(meta['subject']) if meta['subject'].ndim == 1 else -1
    if (n != 60 or any(meta[k].shape != (n,) for k in META)
            or any(meta[k].dtype.kind not in 'iu' for k in ('session', 'activity_id', 'window_index'))
            or any(meta[k].dtype.kind not in 'US' for k in ('subject', 'role'))):
        raise ValueError('Exact sixty-row metadata schema required')
    subject, role = meta['subject'].astype(str), meta['role'].astype(str)
    if set(subject) != set(ACCOUNTS):
        raise PermissionError('Only both designated selection accounts permitted')
    for session, actual_role in zip(meta['session'], role):
        expected = 'train_enrollment' if 1 <= session <= 8 else 'train_selection'
        if not 1 <= session <= 16 or actual_role != expected:
            raise PermissionError('Unauthorized selection session/role')
    expected_counts = {'180679': (5, 3), '622852': (10, 42)}
    for sid, counts in expected_counts.items():
        if tuple(int(np.sum((subject == sid) & (role == r))) for r in
                 ('train_enrollment', 'train_selection')) != counts:
            raise ValueError('Frozen original per-account window counts changed')
    if len(set(zip(*(meta[k].tolist() for k in META)))) != n:
        raise ValueError('Duplicate original window metadata')
    return meta


def load_paired(features, tap_features):
    with np.load(features, allow_pickle=False) as base, np.load(tap_features, allow_pickle=False) as tap:
        meta = validate_metadata({k: base[k].copy() for k in META})
        for k in META:
            if not np.array_equal(meta[k], tap['window_' + k]):
                raise ValueError('Tap/base window metadata mismatch: ' + k)
        key, imu = base['key'].copy(), base['imu'].copy()
        taps, index = validate_taps(tap['tap_vectors'], tap['tap_window'], 60)
        counts = np.bincount(index, minlength=60)
        if (not np.array_equal(counts, tap['window_tap_count'])
                or not np.array_equal(counts >= 5, tap['window_scorable'])
                or tap['window_scorable'].dtype != np.bool_):
            raise ValueError('Stored tap coverage disagrees with actual counts')
    if (key.shape != (60, 50, 10) or imu.shape != (60, 100, 24)
            or key.dtype != np.float32 or imu.dtype != np.float32
            or not np.isfinite(key).all() or not np.isfinite(imu).all()):
        raise ValueError('Frozen encoder feature schema mismatch')
    return meta, key, imu, taps, index


def personal_galleries(meta, taps, index):
    counts = np.bincount(index, minlength=len(meta['subject']))
    gallery = (meta['role'].astype(str) == 'train_enrollment') & (counts >= 5)
    profiles, diagnostics = build_profiles(taps, index, meta['subject'], gallery, ACCOUNTS)
    missing = [a for a in ACCOUNTS if a not in profiles]
    return gallery, profiles, diagnostics, missing


def paired_scores(embeddings, meta, taps, index, gallery, profiles, frozen):
    """All60×2 original claims, immutable scales and frozen thresholds external."""
    if set(profiles) != set(ACCOUNTS):
        raise ValueError('Both personal galleries required')
    tap = score_windows(taps, index, len(meta['subject']), profiles, ACCOUNTS)
    scores, available = {'tap11': tap['distances']}, {'tap11': tap['available']}
    for branch in BRANCHES:
        centers = mean_gallery(embeddings[branch][gallery], meta['subject'][gallery])
        accounts, scores[branch] = euclidean_scores(embeddings[branch], centers)
        if accounts != ACCOUNTS: raise ValueError('Encoder gallery account mismatch')
        available[branch] = np.isfinite(scores[branch])
    for method, branches in FUSIONS.items():
        scores[method], available[method] = fuse(
            np.stack([scores[b] for b in branches], -1),
            np.stack([available[b] for b in branches], -1),
            np.asarray(frozen['scales'][method]))
    return scores, available


def validate_frozen_parameters(frozen):
    if frozen.get('status') != 'complete' or set(frozen.get('thresholds', {})) != set(METHODS):
        raise ValueError('Complete six-method frozen TRAIN parameters required')
    if set(frozen.get('scales', {})) != set(FUSIONS):
        raise ValueError('Both fixed fusion scales required')
    for name, branches in FUSIONS.items():
        scale = np.asarray(frozen['scales'][name], dtype=float)
        if scale.shape != (len(branches),) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid frozen scale')
    for thresholds in frozen['thresholds'].values():
        if (set(thresholds) != {'0.001', '0.01', '0.05'}
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in thresholds.values())):
            raise ValueError('Invalid frozen thresholds')


def prepare(args):
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get('methods') != list(METHODS) or protocol.get('identities') != ACCOUNTS
            or protocol.get('operating_target') != .01
            or not protocol.get('status', '').startswith('Frozen root-approved')):
        raise ValueError('Reviewed selection protocol required')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected: raise ValueError('Protocol source/input changed')
    if digest(args.features) != protocol['selection_features']['sha256']:
        raise ValueError('Wrong original selection archive')
    tap_report = json.loads(args.tap_report.read_text())
    if (tap_report.get('feature_archive_sha256') != digest(args.tap_features)
            or tap_report.get('held_out_measurements_read') is not False
            or tap_report.get('plan', {}).get('proposal_sha256') != digest(args.protocol)
            or tap_report.get('plan', {}).get('selection_window_archive_sha256') != digest(args.features)):
        raise ValueError('Tap extraction is not bound to original selection windows')
    paired_path = args.paired_dir / 'paired_train_complete.json'
    expected_paired = protocol['source_hashes']['research/benchmarks/hmog/tap_paired_train_v2/paired_train_complete.json']
    if digest(paired_path) != expected_paired: raise ValueError('Wrong paired TRAIN parameters')
    frozen = json.loads(paired_path.read_text()); validate_frozen_parameters(frozen)
    selected_path = args.selection_dir / 'selection_complete.json'
    selection = json.loads(selected_path.read_text())
    if (selection.get('status') != 'complete' or selection.get('selected_epoch') != 3
            or selection.get('selected_checkpoint_sha256') != frozen['plan']['checkpoint_sha256']
            or digest(args.selection_dir / 'selected_checkpoint.pt') != frozen['plan']['checkpoint_sha256']):
        raise ValueError('Exact frozen epoch3 checkpoint required')
    inputs = [args.features, args.tap_features, args.tap_report, args.protocol, paired_path,
              selected_path, args.selection_dir / 'selected_checkpoint.pt']
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'canonical_paths': canonical_paths(args),
            'input_sha256': {str(p.resolve().relative_to(ROOT)): digest(p) for p in inputs},
            'source_sha256': {s: digest(ROOT / 'scripts' / s) for s in SOURCES},
            'thresholds': frozen['thresholds'], 'scales': frozen['scales'],
            'checkpoint_sha256': frozen['plan']['checkpoint_sha256'],
            'methods': list(METHODS), 'accounts': ACCOUNTS, 'batch_size': 8, 'cpu_threads': 2,
            'scope': 'Selection ranking only; personal galleries, no global refit/calibration/DEV/test',
            'caveats': protocol['caveats']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'selection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for name in SOURCES:
        (args.output / ('source_' + name)).write_bytes((ROOT / 'scripts' / name).read_bytes())
    print('Prepared only; no selection feature values read')


def run(args):
    plan = json.loads((args.output / 'selection_plan.json').read_text())
    if canonical_paths(args) != plan['canonical_paths']:
        raise ValueError('CLI paths differ from prepared plan')
    for relative, expected in plan['input_sha256'].items():
        if digest(ROOT / relative) != expected: raise ValueError('Frozen input changed')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != expected: raise ValueError('Frozen source changed')
    # Exclusive run marker prevents implicit retry/overwrite of partial results.
    (args.output / 'run_started.json').open('x').close()
    meta, key, imu, taps, index = load_paired(args.features, args.tap_features)
    gallery, profiles, profile_diagnostics, missing = personal_galleries(meta, taps, index)
    result = {'plan': plan, 'profile_diagnostics': profile_diagnostics,
              'missing_gallery_accounts': missing, 'dev_or_test_read': False}
    if missing:
        result.update(status='infeasible', selected_method=None, reason='Both personal accounts must enroll')
        (args.output / 'selection_complete.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True); torch.manual_seed(20260920)
    cls, _ = load_author_classes(); model = cls(10, 24, 50, 100, 64).eval()
    load_checkpoint(model, args.selection_dir / 'selected_checkpoint.pt', plan['checkpoint_sha256'])
    scores, available = paired_scores(embed_all(model, key, imu), meta, taps, index,
                                      gallery, profiles, plan)
    probe = meta['role'].astype(str) == 'train_selection'
    summary = summarize(scores, available, meta['subject'], ACCOUNTS, probe, plan['thresholds'])
    result.update(summary)
    arrays = {**meta, 'accounts': np.array(ACCOUNTS), 'gallery_mask': gallery, 'probe_mask': probe,
              **{'distance_' + k: v for k, v in scores.items()},
              'shared_intersection': np.logical_and.reduce([available[m] for m in METHODS]),
              **{'available_' + k: v for k, v in available.items()}}
    with (args.output / 'selection_scores.npz').open('xb') as stream: np.savez_compressed(stream, **arrays)
    (args.output / 'profiles.json').write_text(json.dumps(profiles, indent=2, allow_nan=False) + '\n')
    for relative, expected in plan['input_sha256'].items():
        if digest(ROOT / relative) != expected: raise ValueError('Frozen input changed during scoring')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != expected: raise ValueError('Frozen source changed during scoring')
    result.update(scores_sha256=digest(args.output / 'selection_scores.npz'),
                  profiles_sha256=digest(args.output / 'profiles.json'))
    (args.output / 'selection_complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'scores_sha256': result['scores_sha256']}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    for name in PATH_ARGS: parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if any(not getattr(args, name).resolve().is_relative_to(ROOT) for name in PATH_ARGS):
        raise ValueError('Repository paths only')
    prepare(args) if args.action == 'prepare' else run(args)


if __name__ == '__main__': main()
