"""Execute unchanged author architecture on generated inputs, never datasets."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'research/benchmarks/references/type2branch'
OUT = ROOT / 'research/benchmarks/type2branch_reference_smoke_v3'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    paths = [REF / 'model.py', REF / 'conf.small.1Kusers.py', Path(__file__),
             ROOT / 'research/benchmarks/type2branch_runtime_v1/keras_compat.json',
             ROOT / '.research-type2branch-deps/tf_keras/src/backend.py']
    plan = {'scope': 'Generated architecture forward/backward only; no Set2Set loss or dataset',
            'seed': 20260920, 'shape': [2, 100, 5], 'cpu_threads': 2,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='',
                      TF_NUM_INTEROP_THREADS='2', TF_NUM_INTRAOP_THREADS='2',
                      TF_CPP_MIN_LOG_LEVEL='2',
                      KERAS_HOME=str(ROOT / '.research-tmp/type2branch-keras'),
                      XDG_CACHE_HOME=str(ROOT / '.research-tmp/type2branch-cache'))
    sys.path.insert(0, str(ROOT / '.research-type2branch-deps'))
    import tensorflow as tf
    import tf_keras
    import numpy as np
    tf.keras.utils.set_random_seed(plan['seed'])
    module('conf', REF / 'conf.small.1Kusers.py')
    author = module('type2branch_author_model', REF / 'model.py')
    built = author.get_model_Type2Branch({'SEQUENCE_LENGTH': 100, 'INPUT_FEATURES': 5})
    model = built['model']
    rng = np.random.default_rng(plan['seed'])
    x = rng.normal(0, .1, size=plan['shape']).astype('float32')
    x[:, :, 0] = rng.integers(0, 256, size=(2, 100)) / 255
    x[:, :, 1:3] = np.abs(x[:, :, 1:3])
    start = time.perf_counter()
    outputs = model(x, training=False)
    assert outputs.shape == (2, 256) and np.isfinite(outputs.numpy()).all()
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with tf.GradientTape() as tape:
        training_output = model(x, training=True)
        diagnostic_loss = tf.reduce_mean(tf.square(training_output))
    gradients = tape.gradient(diagnostic_loss, model.trainable_variables)
    assert all(g is not None for g in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(g)))) for g in gradients)
    backward_seconds = time.perf_counter() - start
    report = {'status': 'generated_forward_backward_passed', 'tensorflow': tf.__version__,
              'tf_keras': tf_keras.__version__,
              'module_paths': {'tensorflow': tf.__file__, 'tf_keras': tf_keras.__file__,
                               'numpy': np.__file__},
              'parameters': model.count_params(), 'trainable_tensors': len(gradients),
              'output_shape': list(outputs.shape), 'forward_seconds': forward_seconds,
              'forward_backward_seconds': backward_seconds,
              'diagnostic_squared_output_loss': float(diagnostic_loss.numpy()),
              'limitations': ['Not original author runtime parity', 'Not Set2Set optimization',
                              'Batch2 diagnostic is not a full batch150 epoch benchmark',
                              'No real feature bridge, dataset, DEV or test access']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
