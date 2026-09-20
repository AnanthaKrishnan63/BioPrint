"""Bounded fit-only BehaveFormer adaptation; no validation or dataset loading.

Accepts a provenance-frozen feature archive. Every epoch is retained so later
TRAIN selection can choose a checkpoint without retraining against DEV results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from hmog_training_core import FIT_IDS, TripletSampler, load_author_classes

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fit_arrays(path):
    """Reject malformed and non-fit data before allocating tensors or fitting."""
    with np.load(path, allow_pickle=False) as archive:
        key = archive['key'].copy()
        imu = archive['imu'].copy()
        subjects = archive['subject'].astype(str)
        sessions = archive['session'].copy()
        roles = archive['role'].astype(str)
    if key.ndim != 3:
        raise ValueError('Keyboard features must be a three-dimensional array')
    n = len(key)
    if (key.shape != (n, 50, 10) or imu.shape != (n, 100, 24)
            or any(x.shape != (n,) for x in (subjects, sessions, roles))
            or sessions.dtype.kind not in 'iu'
            or key.dtype != np.float32 or imu.dtype != np.float32
            or not np.isfinite(key).all() or not np.isfinite(imu).all()):
        raise ValueError('Feature shapes/dtypes/values violate the frozen contract')
    if not set(subjects).issubset(FIT_IDS):
        raise PermissionError('Feature archive contains non-fit identities')
    for session, role in zip(sessions, roles):
        expected = 'train_enrollment' if 1 <= session <= 8 else 'train_fit'
        if not 1 <= session <= 16 or role != expected:
            raise PermissionError('Feature archive contains unauthorized record roles')
    keep = roles == 'train_fit'
    if not keep.any():
        raise ValueError('No population-fit windows')
    return key[keep], imu[keep], subjects[keep], sessions[keep], roles[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--extraction-report', type=Path, required=True)
    parser.add_argument('--model-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.features, args.extraction_report, args.model_protocol, args.output):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('All artifacts must remain inside repository')
    # No overwriting or implicit resume. Failed experiments keep their traces.
    args.output.mkdir(parents=True, exist_ok=False)
    model_class, loss_class = load_author_classes()
    extraction = json.loads(args.extraction_report.read_text())
    if (extraction.get('feature_archive_sha256') != digest(args.features)
            or extraction.get('held_out_measurements_read') is not False):
        raise ValueError('Extraction report must bind exact features and exclude held-out reads')
    frozen = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'features_path': str(args.features.resolve().relative_to(ROOT)),
              'features_sha256': digest(args.features),
              'extraction_report_sha256': digest(args.extraction_report),
              'training_source_sha256': digest(Path(__file__)),
              'core_source_sha256': digest(ROOT / 'scripts/hmog_training_core.py'),
              'model_protocol_path': str(args.model_protocol.resolve().relative_to(ROOT)),
              'model_protocol_sha256': digest(args.model_protocol),
              'seed': 20260920, 'epochs': 20, 'batches_per_epoch': 40,
              'triplets_per_batch': 8, 'cpu_threads': 2,
              'architecture': 'pinned author acc+gyro, 3376714 parameters',
              'optimizer': 'Adam lr0.001, torch defaults',
              'loss': 'pinned author Euclidean triplet hinge margin1; no epsilon',
              'forward': 'three separate anchor/positive/negative calls',
              'population_fit': 'fit identities only; train_fit sessions9–16',
              'normalization': 'source fixed scaling in frozen features; no learned transform',
              'checkpoint_policy': 'retain every epoch; no evaluation or selection in this script',
              'held_out_measurements_accessed': False,
              'adaptations': ['strict recording-time windows', 'window-local IMU operators',
                              'actual observed key pairs', 'reduced training budget']}
    (args.output / 'training_plan.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (args.output / 'training_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'training_core_source.py').write_bytes((ROOT / 'scripts/hmog_training_core.py').read_bytes())
    # Recheck the feature hash after loading and again after fitting.
    key, imu, subjects, sessions, roles = load_fit_arrays(args.features)
    if digest(args.features) != frozen['features_sha256']:
        raise ValueError('Feature archive changed while loading')
    sampler = TripletSampler(subjects, sessions, roles, seed=frozen['seed'])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(frozen['seed'])
    model = model_class(10, 24, 50, 100, 64).train()
    assert sum(p.numel() for p in model.parameters()) == 3376714
    loss_fn = loss_class(margin=1.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    tensors = (torch.from_numpy(key), torch.from_numpy(imu))
    started = time.perf_counter()
    checkpoints = []
    with (args.output / 'history.jsonl').open('x') as history:
        for epoch in range(1, 21):
            losses = []
            for _ in range(40):
                indices = sampler.sample(8)
                optimizer.zero_grad(set_to_none=True)
                embeddings = [model([x[indices[:, part]] for x in tensors]) for part in range(3)]
                loss = loss_fn(*embeddings)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError('Nonfinite loss or gradients; no silent objective changes')
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite parameters')
                losses.append(float(loss.detach()))
            checkpoint = args.output / f'epoch_{epoch:02d}.pt'
            torch.save(model.state_dict(), checkpoint)
            row = {'epoch': epoch, 'mean_fit_loss': float(np.mean(losses)),
                   'elapsed_seconds': time.perf_counter() - started,
                   'checkpoint': checkpoint.name, 'sha256': digest(checkpoint)}
            checkpoints.append(row)
            history.write(json.dumps(row) + '\n')
            history.flush()
            print(json.dumps(row), flush=True)
    if digest(args.features) != frozen['features_sha256']:
        raise ValueError('Feature archive changed during fitting')
    report = {'plan': frozen, 'fit_windows': len(key),
              'fit_windows_by_subject': {s: int(sum(subjects == s)) for s in sorted(set(subjects))},
              'checkpoints': checkpoints, 'elapsed_seconds': time.perf_counter() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'torch': torch.__version__, 'validation_performed': False,
              'all_parameters_finite': True}
    (args.output / 'training_complete.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
