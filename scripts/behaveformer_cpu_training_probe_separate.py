"""Separate-forward synthetic CPU training-cost probe of pinned author BehaveFormer code.

No datasets or checkpoints are opened and no trained weights are persisted.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch

SOURCE = ROOT / 'research/benchmarks/references/hmog/behaveformer-experiments_keystroke_imu_combined_HMOGDB_imu_acc_gyr_model.py'
LOSS_SOURCE = ROOT / 'research/benchmarks/references/hmog/behaveformer-experiments_keystroke_imu_combined_HMOGDB_imu_acc_gyr_train.py'
RECEIPT = ROOT / 'research/benchmarks/references/hmog/source_receipt.json'
OUT = ROOT / 'research/benchmarks/references/hmog/behaveformer_cpu_training_probe_separate.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if OUT.exists():
        raise SystemExit('Preserve existing hardware probe')
    receipt = json.loads(RECEIPT.read_text())
    for path in [SOURCE, LOSS_SOURCE]:
        if digest(path) != receipt['files'][path.name]:
            raise ValueError('Pinned author source changed')
    # Import only the architecture, whose module defines classes without I/O.
    spec = importlib.util.spec_from_file_location('author_behaveformer_cpu_probe', SOURCE)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    # Compile only the exact author loss class. Importing its full train module
    # would also import data/download helpers, which this probe must not invoke.
    import ast
    tree = ast.parse(LOSS_SOURCE.read_text())
    loss_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TripletLoss')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[loss_node], type_ignores=[]), str(LOSS_SOURCE), 'exec'), namespace)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20260920)
    torch.use_deterministic_algorithms(True)
    model = author.Model(10, 24, 50, 100, 64).train()
    loss_function = namespace['TripletLoss'](margin=1.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # Generated tensors are not physiological simulations or accuracy evidence.
    inputs = [torch.randn(24, 50, 10), torch.randn(24, 100, 24)]
    samples = []
    all_finite = True
    started = time.perf_counter()
    for index in range(13):
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        anchor = model([x[:8] for x in inputs])
        positive = model([x[8:16] for x in inputs])
        negative = model([x[16:24] for x in inputs])
        loss = loss_function(anchor, positive, negative)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients)
        if not finite:
            raise FloatingPointError('Nonfinite synthetic loss or gradients')
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError('Nonfinite parameter after update')
        elapsed = time.perf_counter() - begin
        all_finite &= finite
        samples.append({'index': index, 'warmup': index < 3, 'seconds': elapsed,
                        'loss': float(loss.detach()), 'gradient_tensors': len(gradients),
                        'all_loss_gradients_parameters_finite': finite})
    measured = [sample['seconds'] for sample in samples if not sample['warmup']]
    median = float(np.median(measured))
    report = {'scope': 'Generated tensors only; no real measurements, checkpoints, identity selection or accuracy evaluation',
              'source_repository': receipt['author_repository'], 'source_commit': receipt['commit'],
              'architecture_sha256': digest(SOURCE), 'loss_source_sha256': digest(LOSS_SOURCE),
              'probe_sha256': digest(Path(__file__)), 'seed': 20260920,
              'torch': torch.__version__, 'cpu_threads': 2, 'interop_threads': 2,
              'parameters': sum(p.numel() for p in model.parameters()),
              'triplets_per_step': 8, 'examples_per_step': 24,
              'forward_calls_per_step': 3, 'examples_per_forward': 8,
              'batchnorm_convention': 'author separate anchor/positive/negative forward calls; running statistics update three times per optimizer step',
              'key_shape': [24, 50, 10], 'imu_shape': [24, 100, 24],
              'loss': 'exact author unsquared Euclidean triplet loss; margin1',
              'optimizer': 'Adam lr0.001, PyTorch defaults', 'warmup_steps': 3, 'measured_steps': 10,
              'median_step_seconds': median, 'p95_step_seconds': float(np.quantile(measured, .95)),
              'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
              'all_finite': all_finite, 'samples': samples,
              'elapsed_seconds': time.perf_counter() - started,
              'rough_compute_extrapolation': {'epochs': 20, 'batches_per_epoch': 40,
                    'optimizer_steps': 800, 'seconds_at_median': 800 * median,
                    'minutes_at_median': 800 * median / 60,
                    'caveat': 'Illustrative compute only for8triplets perstep; excludes preprocessing, validation, I/O and contention changes. Author separate-forward execution but reduced8triplet batch; not full64triplet schedule or an accuracy prediction.'},
              'weights_persisted': False, 'dataset_accessed': False}
    with OUT.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    run()
