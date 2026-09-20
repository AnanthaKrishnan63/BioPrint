"""Frozen six-method DEV negative comparison; prepare/run, no raw loaders.

No model selection, global fitting, threshold calibration or promotion. Missing
personal galleries stop scoring; missing genuine-probe accounts remain explicit.
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
from hmog_tap_selection import validate_frozen_parameters

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ['219303', '556357', '737973', '777078']
META = ('subject', 'session', 'role', 'activity_id', 'window_index')
METHODS = ('tap11', 'key', 'imu', 'joint', 'key_imu_tap11', 'joint_tap11')
BRANCHES = ('key', 'imu', 'joint')
FUSIONS = {'joint_tap11': ('joint', 'tap11'), 'key_imu_tap11': ('key', 'imu', 'tap11')}
PATH_ARGS = ('features', 'tap_features', 'tap_report', 'paired_dir', 'selection_dir',
             'failure_report', 'protocol', 'output')
SOURCES = ('hmog_tap_dev.py', 'hmog_tap_dev_metrics.py', 'hmog_tap_selection.py',
           'hmog_tap_selection_metrics.py', 'hmog_select.py', 'hmog_training_core.py',
           'hmog_verification.py', 'hmog_tap_benchmark.py', 'hmog_tap_reference.py', 'hmog_score_fusion.py')
COUNTS = {'219303': (24, 7), '556357': (1, 0), '737973': (1, 0), '777078': (11, 24)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_paths(args):
    return {name: str(getattr(args, name).resolve()) for name in PATH_ARGS}


def validate_metadata(meta):
    if (any(meta[k].shape != (68,) for k in META)
            or any(meta[k].dtype.kind not in 'iu' for k in ('session', 'activity_id', 'window_index'))
            or any(meta[k].dtype.kind not in 'US' for k in ('subject', 'role'))):
        raise ValueError('Exact68-row DEV metadata schema required')
    subject, role = meta['subject'].astype(str), meta['role'].astype(str)
    if set(subject) != set(ACCOUNTS): raise PermissionError('Only four designated DEV accounts')
    for session, actual in zip(meta['session'], role):
        expected = 'train_support' if 1 <= session <= 8 else 'dev_probe'
        if not 1 <= session <= 16 or actual != expected:
            raise PermissionError('Unauthorized DEV record role/session')
    for sid, expected in COUNTS.items():
        counts = tuple(int(np.sum((subject == sid) & (role == r))) for r in ('train_support', 'dev_probe'))
        if counts != expected: raise ValueError('Original per-account coverage changed')
    if len(set(zip(*(meta[k].tolist() for k in META)))) != 68:
        raise ValueError('Duplicate original DEV window')
    return meta


def load_paired(features, tap_features):
    with np.load(features, allow_pickle=False) as base, np.load(tap_features, allow_pickle=False) as tap:
        meta = validate_metadata({k: base[k].copy() for k in META})
        for name in META:
            if not np.array_equal(meta[name], tap['window_' + name]):
                raise ValueError('Tap/base metadata mismatch: ' + name)
        key, imu = base['key'].copy(), base['imu'].copy()
        taps, index = validate_taps(tap['tap_vectors'], tap['tap_window'], 68)
        counts = np.bincount(index, minlength=68)
        if (not np.array_equal(counts, tap['window_tap_count'])
                or not np.array_equal(counts >= 5, tap['window_scorable'])
                or tap['window_scorable'].dtype != np.bool_):
            raise ValueError('Tap missingness/count mismatch')
    if (key.shape != (68, 50, 10) or imu.shape != (68, 100, 24)
            or key.dtype != np.float32 or imu.dtype != np.float32
            or not np.isfinite(key).all() or not np.isfinite(imu).all()):
        raise ValueError('Invalid DEV feature schema')
    return meta, key, imu, taps, index


def personal_galleries(meta, taps, index):
    counts = np.bincount(index, minlength=68)
    gallery = (meta['role'].astype(str) == 'train_support') & (counts >= 5)
    profiles, diagnostics = build_profiles(taps, index, meta['subject'], gallery, ACCOUNTS)
    missing = [a for a in ACCOUNTS if a not in profiles]
    return gallery, profiles, diagnostics, missing


def coverage(meta, index, gallery):
    counts = np.bincount(index, minlength=68)
    result = {}
    for sid in ACCOUNTS:
        result[sid] = {}
        for role in ('train_support', 'dev_probe'):
            mask = (meta['subject'].astype(str) == sid) & (meta['role'].astype(str) == role)
            result[sid][role] = {'original_windows': int(mask.sum()),
                                'tap_scorable_windows': int(np.sum(mask & (counts >= 5))),
                                'tap_vectors': int(counts[mask].sum())}
    return {'accounts': result, 'original_windows': 68, 'support_windows': 37, 'probe_windows': 31,
            'shared_gallery_windows': int(gallery.sum()),
            'scope': 'Existing original68windows only; upstream extraction/session losses remain separate'}


def paired_scores(embeddings, meta, taps, index, gallery, profiles, frozen):
    if set(profiles) != set(ACCOUNTS): raise ValueError('All four galleries required')
    tap = score_windows(taps, index, 68, profiles, ACCOUNTS)
    scores, available = {'tap11': tap['distances']}, {'tap11': tap['available']}
    for branch in BRANCHES:
        centers = mean_gallery(embeddings[branch][gallery], meta['subject'][gallery])
        accounts, scores[branch] = euclidean_scores(embeddings[branch], centers)
        if accounts != ACCOUNTS: raise ValueError('Missing encoder personal gallery')
        available[branch] = np.isfinite(scores[branch])
    for method, branches in FUSIONS.items():
        scores[method], available[method] = fuse(np.stack([scores[b] for b in branches], -1),
            np.stack([available[b] for b in branches], -1), np.asarray(frozen['scales'][method]))
    return scores, available


def upstream_candidate_count(report):
    coverage = report.get('coverage', {})
    if set(coverage) != set(ACCOUNTS): raise ValueError('Four-account upstream coverage required')
    counts = [coverage[a]['dev_probe']['candidate_windows'] for a in ACCOUNTS]
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 154:
        raise ValueError('Expected154 known inspected candidate probes; excluded counts remain unknown')
    return sum(counts)


def prepare(args):
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get('evaluation_only') is not True or protocol.get('no_method_ranking_or_promotion') is not True
            or protocol.get('methods') != list(METHODS) or 'PROPOSAL' in protocol.get('status', '').upper()):
        raise ValueError('Final reviewed evaluation-only protocol required')
    for record in protocol['bindings'].values():
        if digest(ROOT / record['path']) != record['sha256']: raise ValueError('Protocol-bound input changed')
    expected_paths = {'dev_window_archive': args.features,
                      'encoder_checkpoint': args.selection_dir / 'selected_checkpoint.pt',
                      'fit_parameters': args.paired_dir / 'paired_train_complete.json',
                      'selection_report': args.failure_report}
    for name, path in expected_paths.items():
        if path.resolve() != (ROOT / protocol['bindings'][name]['path']).resolve():
            raise ValueError('CLI artifact path differs from protocol binding')
    extraction = json.loads(args.tap_report.read_text())
    if (extraction.get('feature_archive_sha256') != digest(args.tap_features)
            or extraction.get('plan', {}).get('proposal_sha256') != digest(args.protocol)
            or extraction.get('plan', {}).get('dev_window_archive_sha256') != digest(args.features)
            or extraction.get('test_measurements_read') is not False):
        raise ValueError('DEV tap extraction provenance mismatch')
    upstream = ROOT / 'research/benchmarks/hmog/validation_v1/validation_complete.json'
    if (extraction.get('plan', {}).get('upstream_coverage_path') != str(upstream.relative_to(ROOT))
            or extraction.get('plan', {}).get('upstream_coverage_sha256') != digest(upstream)):
        raise ValueError('Upstream coverage binding mismatch')
    upstream_report = json.loads(upstream.read_text())
    expected_candidates = upstream_candidate_count(upstream_report)
    frozen = json.loads(expected_paths['fit_parameters'].read_text()); validate_frozen_parameters(frozen)
    if frozen['plan']['checkpoint_sha256'] != digest(expected_paths['encoder_checkpoint']):
        raise ValueError('Checkpoint differs from frozen paired parameters')
    failure = json.loads(args.failure_report.read_text())
    if failure.get('status') != 'no_useful_discrimination':
        raise ValueError('Preserve original failed-selection outcome')
    inputs = [args.protocol, args.features, args.tap_features, args.tap_report, args.failure_report,
              expected_paths['fit_parameters'], expected_paths['encoder_checkpoint'], upstream]
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'canonical_paths': canonical_paths(args),
            'input_sha256': {str(p.resolve().relative_to(ROOT)): digest(p) for p in inputs},
            'source_sha256': {s: digest(ROOT / 'scripts' / s) for s in SOURCES},
            'checkpoint_sha256': frozen['plan']['checkpoint_sha256'],
            'thresholds': frozen['thresholds'], 'scales': frozen['scales'],
            'accounts': ACCOUNTS, 'methods': list(METHODS), 'evaluation_only': True,
            'no_method_ranking_or_promotion': True, 'selection_status': failure['status'],
            'cpu_threads': 2, 'batch_size': 8, 'scope_caveats': protocol['scope_caveats'],
            'known_inspected_candidate_probes': expected_candidates,
            'upstream_coverage': upstream_report['coverage']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'dev_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for name in SOURCES:
        (args.output / ('source_' + name)).write_bytes((ROOT / 'scripts' / name).read_bytes())
    print('Prepared only; no DEV feature values read')


def run(args):
    plan = json.loads((args.output / 'dev_plan.json').read_text())
    if canonical_paths(args) != plan['canonical_paths']: raise ValueError('CLI paths differ from frozen plan')
    for relative, sha in plan['input_sha256'].items():
        if digest(ROOT / relative) != sha: raise ValueError('Frozen input changed')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != sha: raise ValueError('Frozen source changed')
    (args.output / 'run_started.json').open('x').close()
    meta, key, imu, taps, index = load_paired(args.features, args.tap_features)
    gallery, profiles, diagnostics, missing = personal_galleries(meta, taps, index)
    result = {'plan': plan, 'coverage': coverage(meta, index, gallery), 'profile_diagnostics': diagnostics,
              'missing_gallery_accounts': missing, 'missing_probe_accounts': ['556357', '737973'],
              'evaluation_only': True, 'no_method_ranking_or_promotion': True, 'test_measurements_read': False}
    if missing:
        result.update(status='infeasible', reason='Missing required personal support gallery', metrics={})
        (args.output / 'dev_complete.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True); torch.manual_seed(20260920)
    cls, _ = load_author_classes(); model = cls(10, 24, 50, 100, 64).eval()
    load_checkpoint(model, args.selection_dir / 'selected_checkpoint.pt', plan['checkpoint_sha256'])
    scores, available = paired_scores(embed_all(model, key, imu), meta, taps, index, gallery, profiles, plan)
    from hmog_tap_dev_metrics import summarize
    probe = meta['role'].astype(str) == 'dev_probe'
    result.update(summarize(scores, available, meta['subject'], ACCOUNTS, probe, plan['thresholds'],
                            expected_candidate_probes=plan['known_inspected_candidate_probes']))
    arrays = {**meta, 'accounts': np.array(ACCOUNTS), 'gallery_mask': gallery, 'probe_mask': probe,
              'shared_intersection': np.logical_and.reduce([available[m] for m in METHODS]),
              **{'distance_' + k: v for k, v in scores.items()},
              **{'available_' + k: v for k, v in available.items()}}
    with (args.output / 'dev_scores.npz').open('xb') as stream: np.savez_compressed(stream, **arrays)
    (args.output / 'profiles.json').write_text(json.dumps(profiles, indent=2, allow_nan=False) + '\n')
    for relative, sha in plan['input_sha256'].items():
        if digest(ROOT / relative) != sha: raise ValueError('Frozen input changed during comparison')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / 'scripts' / name) != sha: raise ValueError('Frozen source changed during comparison')
    result.update(scores_sha256=digest(args.output / 'dev_scores.npz'), profiles_sha256=digest(args.output / 'profiles.json'))
    (args.output / 'dev_complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'missing_gallery_accounts': missing,
                      'missing_probe_accounts': result['missing_probe_accounts']}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    for name in PATH_ARGS: parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if any(not getattr(args, n).resolve().is_relative_to(ROOT) for n in PATH_ARGS): raise ValueError('Repository paths only')
    prepare(args) if args.action == 'prepare' else run(args)


if __name__ == '__main__': main()
