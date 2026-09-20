"""Independently recompute operating counts and check artifact integrity."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import joblib
from infer import score_input

HERE = Path(__file__).parent
OUT = HERE / 'results'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def check_counts(row, path):
    with np.load(path, allow_pickle=False) as data:
        y, s = data['genuine'], data['scores']
        assert y.dtype == bool and np.isfinite(s).all()
        accepted = s >= row['threshold']
        assert int((accepted & ~y).sum()) == row['false_accepts']
        assert int((~accepted & y).sum()) == row['false_rejects']
        assert int(y.sum()) == row['genuine_attempts']
        assert int((~y).sum()) == row['impostor_attempts']


def main():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    report = json.loads((OUT / 'report.json').read_text())
    plan = json.loads((OUT / 'plan.json').read_text())
    for key, path in [('protocol_sha256', HERE / 'PROTOCOL.md'), ('script_sha256', HERE / 'run.py'),
                      ('engine_sha256', HERE.parents[1] / 'code/bioprint/engine/general_typing.py')]:
        assert digest(path) == plan[key]
    for item in frozen.values():
        assert digest(OUT / item['file']) == item['sha256']
        assert item['calibration']['far'] <= .01
    for key, row in report['results'].items():
        check_counts(row, OUT / f'{key.replace("/", "-")}-scores.npz')
        track = key.split('/')[0]
        assert not set(row['subjects']) & set(report['fit_profiles'][track])
        assert not set(row['subjects']) & set(report['calibration_profiles'][track])
    beacon = json.loads((OUT / 'beacon.json').read_text())
    for key, row in beacon['results'].items():
        check_counts(row, OUT / f'beacon-{key.replace("/", "-")}-scores.npz')
    # Generated browser-format recordings check the learned inference path,
    # not biometric accuracy. Password key sequence is unrelated to CMU.
    sample = {'keystrokes': [e for i, code in enumerate(['KeyA', 'KeyX', 'Digit4', 'Period', 'KeyB'])
              for e in ({'code': code, 'type': 'down', 't': i * 130., 'field': 'password'},
                        {'code': code, 'type': 'up', 't': i * 130. + 80., 'field': 'password'})]}
    payload = {'enrollment': [sample] * 10, 'attempt': sample}
    latency = {}
    for name in ('distance', 'logistic', 'extra_trees'):
        artifact = joblib.load(OUT / frozen[f'pooled/{name}']['file'])
        score_input(payload, artifact)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = score_input(payload, artifact)
            times.append((time.perf_counter() - start) * 1000)
            assert np.isfinite(result['score'])
        latency[name] = {'median_ms': float(np.median(times)), 'max_ms': max(times),
                         'model_bytes': (OUT / frozen[f'pooled/{name}']['file']).stat().st_size}
    result = {'verified_score_tables': len(report['results']) + len(beacon['results']),
              'model_hashes_verified': len(frozen), 'calibration_far_gates_verified': True,
              'source_hashes_verified': True, 'identity_disjointness_verified': True,
              'generated_browser_input_inference': latency}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
