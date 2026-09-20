"""Execute publisher loss/distance ASTs against independent generated oracles.

Only selected definitions are loaded; no Addons package shim, dataset import,
or modification to either publisher function body is used.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/type2branch_loss_audit_v1'


def definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing reviewed publisher definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def oracle(x, n, k, beta, np):
    x = np.asarray(x, dtype=np.float64)
    total, count = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            for a in range(n):
                for b in range(a + 1, n):
                    positive = np.linalg.norm(x[i*n+a] - x[i*n+b])
                    for c in range(n):
                        negative = np.linalg.norm(x[i*n+a] - x[j*n+c])
                        total += max(positive - negative + 1.5, 0)
                        count += 1
    radii = []
    for i in range(k):
        block = x[i*n:(i+1)*n]
        radii.append(np.linalg.norm(block - block.mean(axis=0), axis=1).mean())
    penalty = beta * np.mean(np.abs(np.array(radii)/np.mean(radii) - 1))
    return total/count + penalty


def main():
    OUT.mkdir(exist_ok=False)
    loss_path = ROOT / 'research/benchmarks/references/type2branch/loss.py'
    distance_path = ROOT / 'research/benchmarks/references/type2branch_addons/metric_learning.py'
    sources = [loss_path, distance_path, Path(__file__),
               ROOT / '.research-type2branch-deps/tf_keras/src/backend.py']
    plan = {'scope': 'Generated numerical validation; no observations or optimizer training',
            'seed': 20260920, 'finite_difference_epsilon': .001,
            'gradient_absolute_tolerance': .003,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='',
                      TF_NUM_INTEROP_THREADS='2', TF_NUM_INTRAOP_THREADS='2',
                      TF_CPP_MIN_LOG_LEVEL='2',
                      KERAS_HOME=str(ROOT / '.research-tmp/type2branch-keras'),
                      XDG_CACHE_HOME=str(ROOT / '.research-tmp/type2branch-cache'))
    sys.path.insert(0, str(ROOT / '.research-type2branch-deps'))
    import tensorflow as tf
    import numpy as np
    distance_scope = definitions(distance_path, ['pairwise_distance'],
                                 {'tf': tf, 'TensorLike': tf.Tensor})
    distance = distance_scope['pairwise_distance']
    def make_loss(n, k):
        scope = definitions(loss_path, ['calculate_set2set_loss', 'Set2SetLoss'],
                            {'tf': tf, 'sys': sys, 'Loss': tf.keras.losses.Loss,
                             'conf': types.SimpleNamespace(N=n),
                             'metric_learning': types.SimpleNamespace(pairwise_distance=distance)})
        return scope['Set2SetLoss'](k, .05)
    loss = make_loss(2, 2)
    results = []
    for values, expected_sm in [([0, 2, 4, 8], 0), ([2, 0, 4, 8], .75), ([4, 8, 0, 2], 2.5)]:
        x = np.array(values, dtype=np.float32)[:, None]
        got = float(loss(tf.zeros(4), x).numpy())
        expected = expected_sm + .05/3
        np.testing.assert_allclose(got, expected, atol=2e-6, rtol=0)
        np.testing.assert_allclose(oracle(x, 2, 2, .05, np), expected, atol=1e-12, rtol=0)
        results.append({'ordered_values': values, 'actual': got, 'expected': expected})
    rng = np.random.default_rng(plan['seed'])
    x = rng.normal(size=(4, 3)).astype('float32')
    pairwise = distance(x).numpy()
    np.testing.assert_allclose(pairwise, np.linalg.norm(x[:, None] - x[None, :], axis=-1), atol=1e-6)
    variable = tf.Variable(x)
    with tf.GradientTape() as tape:
        value = loss(tf.zeros(4), variable)
    gradient = tape.gradient(value, variable).numpy()
    expected_value = oracle(x, 2, 2, .05, np)
    np.testing.assert_allclose(float(value.numpy()), expected_value, atol=2e-6)
    numerical = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.astype('float64'), x.astype('float64')
        plus[index] += .001
        minus[index] -= .001
        numerical[index] = (oracle(plus, 2, 2, .05, np) - oracle(minus, 2, 2, .05, np))/.002
    np.testing.assert_allclose(gradient, numerical, atol=.003, rtol=0)
    # Actual published set cardinalities, with generated16D embeddings.
    full = rng.normal(size=(150, 16)).astype('float32')
    full_loss = make_loss(15, 10)
    full_value = float(full_loss(tf.zeros(150), full).numpy())
    full_expected = oracle(full, 15, 10, .05, np)
    np.testing.assert_allclose(full_value, full_expected, atol=2e-5, rtol=0)
    # Preserve rather than fix the source's degenerate behavior.
    collapsed = tf.Variable(np.zeros((4, 3), dtype='float32'))
    with tf.GradientTape() as tape:
        collapsed_value = loss(tf.zeros(4), collapsed)
    collapsed_gradient = tape.gradient(collapsed_value, collapsed).numpy()
    assert not np.isfinite(collapsed_value.numpy())
    report = {'status': 'nondegenerate_oracles_passed_with_source_degeneracy',
              'hand_fixtures': results, 'random_loss_absolute_error': abs(float(value.numpy())-expected_value),
              'gradient_max_absolute_error': float(np.max(np.abs(gradient-numerical))),
              'published_cardinality': {'K': 10, 'N': 15, 'actual': full_value, 'oracle': full_expected},
              'collapsed_loss_finite': bool(np.isfinite(collapsed_value.numpy())),
              'collapsed_gradients_finite': bool(np.isfinite(collapsed_gradient).all()),
              'dataset_observations_read': False,
              'limitations': ['Selected original definitions executed, not whole Addons distribution',
                              'Collapsed embeddings expose source NaN, not repaired',
                              'No end-to-end optimizer step or model convergence claim']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
