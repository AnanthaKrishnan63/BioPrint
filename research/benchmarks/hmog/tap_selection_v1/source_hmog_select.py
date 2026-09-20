"""Frozen twenty-checkpoint HMOG selection only; never trains or calibrates.

Only the two designated TRAIN-selection accounts are permitted. Branch scores
come from the same joint-selected checkpoint, not independent ablation tuning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from hmog_training_core import load_author_classes, MODEL_HASH, LOSS_HASH
from hmog_verification import branch_embeddings, mean_gallery, euclidean_scores, pooled_eer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/benchmarks/hmog/model_protocol_v2.json'
SELECTION_IDS = frozenset(('180679', '622852'))
BRANCHES = ('joint', 'key', 'imu')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), digest_bytes(raw)


def load_selection_arrays(path):
    """Validate metadata before copying measurements from a pickle-free NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        subjects, sessions, roles = (archive[name].copy() for name in ('subject', 'session', 'role'))
        if (subjects.ndim != 1 or subjects.dtype.kind not in 'US'
                or roles.shape != subjects.shape or roles.dtype.kind not in 'US'
                or sessions.shape != subjects.shape or sessions.dtype.kind not in 'iu'):
            raise ValueError('Invalid selection metadata shape/dtype')
        subjects, roles = subjects.astype(str), roles.astype(str)
        if not set(subjects).issubset(SELECTION_IDS):
            raise PermissionError('Non-selection identity is forbidden')
        for session, role in zip(sessions, roles):
            expected = 'train_enrollment' if 1 <= session <= 8 else 'train_selection'
            if not 1 <= session <= 16 or role != expected:
                raise PermissionError('Unauthorized selection session/role')
        key, imu = archive['key'].copy(), archive['imu'].copy()
    n = len(subjects)
    if (key.shape != (n, 50, 10) or imu.shape != (n, 100, 24)
            or key.dtype != np.float32 or imu.dtype != np.float32
            or not np.isfinite(key).all() or not np.isfinite(imu).all()):
        raise ValueError('Invalid selection feature shape/dtype/value')
    return key, imu, subjects, sessions, roles


def cohort_feasibility(subjects, roles):
    missing = []
    for subject in sorted(SELECTION_IDS):
        for role in ('train_enrollment', 'train_selection'):
            if not np.any((subjects == subject) & (roles == role)):
                missing.append({'subject': subject, 'missing_role': role})
    return missing


def coverage_from_report(report, subjects, sessions):
    """Count inspected windows, preserving metadata exclusions with unknown counts."""
    reports = report.get('subjects', {})
    exclusions = report.get('excluded_sessions', {})
    if set(reports) != SELECTION_IDS:
        raise ValueError('Coverage report must describe exactly both selection accounts')
    if not set(exclusions).issubset(SELECTION_IDS):
        raise ValueError('Unknown account in excluded session metadata')
    result = {}
    for subject in sorted(SELECTION_IDS):
        excluded = exclusions.get(subject, {})
        inspected = reports[subject]
        if (set(inspected) & set(excluded)
                or set(inspected) | set(excluded) != {str(x) for x in range(1, 17)}):
            raise ValueError('Coverage must account for all sessions1–16 without overlap')
        for session, record in excluded.items():
            if (not isinstance(record, dict)
                    or record.get('reason') not in ('missing_keypress', 'empty_keypress')
                    or record.get('basis') != 'archive_metadata'
                    or record.get('candidate_windows') is not None
                    or record.get('eligible_windows') is not None
                    or record.get('activity_contents_inspected', False) is not False
                    or np.any((subjects == subject) & (sessions == int(session)))):
                raise ValueError('Excluded session requires metadata-only reason and no features/counts')
        result[subject] = {}
        for role, session_range in [('train_enrollment', range(1, 9)),
                                    ('train_selection', range(9, 17))]:
            candidate = eligible = 0
            for session in session_range:
                if str(session) in excluded:
                    continue
                row = reports[subject][str(session)]
                c, e = row.get('candidate_windows'), row.get('eligible_windows')
                if (type(c) is not int or type(e) is not int or not 0 <= e <= c
                        or e != int(np.sum((subjects == subject) & (sessions == session)))):
                    raise ValueError('Coverage counts do not match selection features')
                candidate += c
                eligible += e
            result[subject][role] = {'candidate_windows': candidate,
                                    'eligible_windows': eligible,
                                    'missing_windows': candidate - eligible,
                                    'coverage': eligible / candidate if candidate else None,
                                    'denominator_scope': 'inspected metadata-nonempty sessions only',
                                    'inspected_sessions': [s for s in session_range if str(s) in inspected],
                                    'excluded_sessions': {str(s): excluded[str(s)] for s in session_range
                                                          if str(s) in excluded},
                                    'excluded_session_candidate_windows': None}
    return result


def validate_training_report(report, protocol_sha):
    plan = report.get('plan', {})
    if (plan.get('model_protocol_sha256') != protocol_sha
            or plan.get('held_out_measurements_accessed') is not False
            or report.get('validation_performed') is not False
            or report.get('all_parameters_finite') is not True
            or (plan.get('epochs'), plan.get('batches_per_epoch'), plan.get('triplets_per_batch')) != (20, 40, 8)):
        raise ValueError('Training report violates frozen protocol')
    rows = report.get('checkpoints', [])
    if len(rows) != 20 or [r.get('epoch') for r in rows] != list(range(1, 21)):
        raise ValueError('Require exactly the twenty ordered frozen checkpoints')
    for row in rows:
        if (row.get('checkpoint') != f"epoch_{row['epoch']:02d}.pt"
                or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('Invalid checkpoint name/hash')
    return rows


def load_checkpoint(model, path, expected_sha):
    """Verify exact bytes before deserialization; no check/reopen race."""
    raw = path.read_bytes()
    if digest_bytes(raw) != expected_sha:
        raise ValueError('Checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or any(not isinstance(x, torch.Tensor) or
                                        not torch.isfinite(x).all() for x in state.values()):
        raise ValueError('Checkpoint must contain only finite state tensors')
    model.load_state_dict(state, strict=True)
    model.eval()


def embed_all(model, key, imu):
    outputs = {b: [] for b in BRANCHES}
    for start in range(0, len(key), 8):
        values = branch_embeddings(model, torch.from_numpy(key[start:start + 8]),
                                    torch.from_numpy(imu[start:start + 8]))
        for branch in BRANCHES:
            outputs[branch].append(values[branch].cpu().numpy())
    return {branch: np.concatenate(values) for branch, values in outputs.items()}


def score_embeddings(embeddings, subjects, roles):
    enrollment, probe = roles == 'train_enrollment', roles == 'train_selection'
    if cohort_feasibility(subjects, roles):
        raise ValueError('Both designated accounts need gallery and probes')
    arrays = {'probe_subject': subjects[probe], 'accounts': np.array(sorted(SELECTION_IDS))}
    metrics = {}
    for branch in BRANCHES:
        gallery = mean_gallery(embeddings[branch][enrollment], subjects[enrollment])
        accounts, scores = euclidean_scores(embeddings[branch][probe], gallery)
        genuine = subjects[probe, None] == np.array(accounts)[None, :]
        arrays[f'{branch}_distances'] = scores
        metrics[branch] = {'pooled_eer': pooled_eer(scores[genuine], scores[~genuine]),
                           'genuine_count': int(genuine.sum()),
                           'impostor_count': int((~genuine).sum())}
    arrays['genuine_mask'] = genuine
    return metrics, arrays


def select_epoch(history):
    if not history or any(not np.isfinite(r['metrics']['joint']['pooled_eer']) for r in history):
        raise ValueError('Finite joint EER history required')
    return min(history, key=lambda r: (r['metrics']['joint']['pooled_eer'], r['epoch']))['epoch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--extraction-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.training_dir, args.features, args.extraction_report, args.output):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('All paths must remain in repository')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, protocol_sha = read_json(PROTOCOL)
    if protocol.get('version') != 2:
        raise ValueError('Expected frozen protocol v2')
    training, training_sha = read_json(args.training_dir / 'training_complete.json')
    checkpoints = validate_training_report(training, protocol_sha)
    extraction, extraction_sha = read_json(args.extraction_report)
    feature_sha = digest_bytes(args.features.read_bytes())
    if (extraction.get('feature_archive_sha256') != feature_sha
            or extraction.get('held_out_measurements_read') is not False):
        raise ValueError('Extraction report does not bind permitted exact features')
    sources = {name: (ROOT / 'scripts' / name).read_bytes() for name in
               ('hmog_select.py', 'hmog_verification.py', 'hmog_training_core.py')}
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_protocol_sha256': protocol_sha, 'training_complete_sha256': training_sha,
            'extraction_report_sha256': extraction_sha, 'features_sha256': feature_sha,
            'sources_sha256': {k: digest_bytes(v) for k, v in sources.items()},
            'author_model_sha256': MODEL_HASH, 'author_loss_sha256': LOSS_HASH,
            'checkpoint_candidates': checkpoints, 'selection_ids': sorted(SELECTION_IDS),
            'objective': 'minimum pooled interpolated joint EER; tie earliest epoch',
            'batch_size': 8, 'cpu_threads': 2, 'calibration_or_dev_accessed': False,
            'branch_caveat': 'Same joint-trained and joint-selected checkpoint ablations'}
    (args.output / 'selection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for name, raw in sources.items():
        (args.output / ('source_' + name)).write_bytes(raw)
    # Plan is durable before any feature values are decompressed or inspected.
    key, imu, subjects, sessions, roles = load_selection_arrays(args.features)
    if digest_bytes(args.features.read_bytes()) != feature_sha:
        raise ValueError('Feature archive changed while loading')
    coverage = coverage_from_report(extraction, subjects, sessions)
    missing = cohort_feasibility(subjects, roles)
    if missing:
        report = {'status': 'infeasible', 'missing': missing, 'coverage': coverage, 'plan': plan}
        (args.output / 'selection_complete.json').write_text(json.dumps(report, indent=2) + '\n')
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260920)
    model_class, _ = load_author_classes()
    model = model_class(10, 24, 50, 100, 64).eval()
    history, best_arrays, best_epoch = [], None, None
    started = time.perf_counter()
    with (args.output / 'score_history.jsonl').open('x') as stream:
        for row in checkpoints:
            path = args.training_dir / row['checkpoint']
            load_checkpoint(model, path, row['sha256'])
            metrics, arrays = score_embeddings(embed_all(model, key, imu), subjects, roles)
            result = {'epoch': row['epoch'], 'checkpoint_sha256': row['sha256'], 'metrics': metrics}
            history.append(result)
            if select_epoch(history) != best_epoch:
                best_epoch, best_arrays = row['epoch'], arrays
            stream.write(json.dumps(result) + '\n'); stream.flush()
            print(json.dumps(result), flush=True)
    chosen = checkpoints[best_epoch - 1]
    chosen_raw = (args.training_dir / chosen['checkpoint']).read_bytes()
    if digest_bytes(chosen_raw) != chosen['sha256']:
        raise ValueError('Chosen checkpoint changed after scoring')
    if digest_bytes(args.features.read_bytes()) != feature_sha:
        raise ValueError('Feature archive changed during selection')
    (args.output / 'selected_checkpoint.pt').write_bytes(chosen_raw)
    with (args.output / 'selected_scores.npz').open('xb') as stream:
        np.savez_compressed(stream, **best_arrays)
    report = {'status': 'complete', 'plan': plan, 'coverage': coverage, 'history': history,
              'selected_epoch': best_epoch, 'selected_checkpoint_sha256': chosen['sha256'],
              'selected_scores_sha256': digest_bytes((args.output / 'selected_scores.npz').read_bytes()),
              'elapsed_seconds': time.perf_counter() - started,
              'selection_only': True, 'calibration_performed': False, 'dev_accessed': False}
    (args.output / 'selection_complete.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
