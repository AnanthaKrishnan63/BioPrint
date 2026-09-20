"""One generated full-cardinality author-model Set2Set optimizer step on CPU."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types

from type2branch_loss_audit import definitions
from type2branch_reference_smoke import module

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'research/benchmarks/references/type2branch'
OUT = ROOT / 'research/benchmarks/type2branch_optimizer_smoke_v1'


def main():
    OUT.mkdir(exist_ok=False)
    distance_path = ROOT / 'research/benchmarks/references/type2branch_addons/metric_learning.py'
    paths = [REF/'model.py', REF/'loss.py', REF/'conf.small.1Kusers.py', distance_path,
             Path(__file__), ROOT/'scripts/type2branch_loss_audit.py',
             ROOT/'scripts/type2branch_reference_smoke.py',
             ROOT/'.research-type2branch-deps/tf_keras/src/backend.py']
    (OUT/'plan.json').write_text(json.dumps({'scope': 'One generated full-size optimizer step',
        'shape': [150, 100, 5], 'K': 10, 'N': 15, 'seed': 20260920, 'cpu_threads': 2,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths}}, indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='',
        TF_NUM_INTEROP_THREADS='2', TF_NUM_INTRAOP_THREADS='2', TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0, str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    import numpy as np
    tf.keras.utils.set_random_seed(20260920)
    conf = module('conf', REF/'conf.small.1Kusers.py')
    author = module('type2branch_author_model', REF/'model.py')
    built = author.get_model_Type2Branch({'SEQUENCE_LENGTH': 100, 'INPUT_FEATURES': 5})
    model = built['model']
    distance = definitions(distance_path, ['pairwise_distance'], {'tf': tf, 'TensorLike': tf.Tensor})
    scope = definitions(REF/'loss.py', ['calculate_set2set_loss', 'Set2SetLoss'],
        {'tf': tf, 'sys': sys, 'Loss': tf.keras.losses.Loss, 'conf': conf,
         'metric_learning': types.SimpleNamespace(pairwise_distance=distance['pairwise_distance'])})
    loss = scope['Set2SetLoss'](conf.K, conf.BETA)
    rng = np.random.default_rng(20260920)
    x = rng.normal(0, .1, size=(150, 100, 5)).astype('float32')
    x[:, :, 0] = rng.integers(0, 256, size=(150, 100))/255
    x[:, :, 1:3] = np.abs(x[:, :, 1:3])
    before = [v.numpy().copy() for v in model.trainable_variables]
    started = time.perf_counter()
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        value = loss(tf.repeat(tf.range(10), 15), predictions)
    gradients = tape.gradient(value, model.trainable_variables)
    if not np.isfinite(value.numpy()) or any(g is None for g in gradients):
        raise ValueError('Nonfinite loss or missing gradient')
    if not all(bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(g)))) for g in gradients):
        raise ValueError('Nonfinite gradient')
    built['optimizer'].apply_gradients(zip(gradients, model.trainable_variables))
    step_seconds = time.perf_counter()-started
    after = [v.numpy() for v in model.trainable_variables]
    assert all(np.isfinite(a).all() for a in after)
    changed = sum(not np.array_equal(a, b) for a,b in zip(before, after))
    assert changed > 0 and int(built['optimizer'].iterations.numpy()) == 1
    report = {'status': 'generated_full_batch_optimizer_step_passed',
        'shape': [150, 100, 5], 'loss': float(value.numpy()),
        'changed_trainable_tensors': changed, 'trainable_tensors': len(after),
        'optimizer_steps': 1, 'step_seconds_including_first_trace': step_seconds,
        'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'dataset_observations_read': False,
        'limitations': ['Generated channels do not establish feature-pipeline fidelity',
                       'One first-traced step is not a steady-state epoch benchmark',
                       'No convergence, DEV, FAR, FRR or EER claim']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
