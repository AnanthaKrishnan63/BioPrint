"""Prepare raw-prefix calibration features using reserved TRAIN identities only."""
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_context_model import MeanContextModel
from type2branch_csv_bridge import integer_rows
from type2branch_random import first_synthesis_thread
from type2branch_variable_adapter import synthesize_window

LENGTHS = (25, 50, 75, 100)
OUT = ROOT/'research/benchmarks/type2branch_length_calibration_features_v1'


def raw_prefix_window(raw, length):
    """Truncate actual raw events before synthesis, retaining signed timings."""
    raw = integer_rows(raw)
    if type(length) is not int or length not in LENGTHS or len(raw) < length:
        raise ValueError('Available raw prefix of 25/50/75/100 events required')
    wire = raw[:length].copy()
    base = wire.astype(np.float64)
    base[:, 0] /= 255.
    base[:, 1:] = np.clip(base[:, 1:]/1000., 0, 30)
    base[0, 2] = 0
    return {'true_length': length, 'raw_ms': wire, 'base': base}


def ordered_indices(subjects, windows, allowed):
    subjects = np.asarray(subjects); windows = np.asarray(windows)
    if (subjects.shape != (240,) or windows.shape != subjects.shape
            or windows.dtype.kind not in 'iu' or len(set(allowed)) != 16
            or set(subjects) != set(allowed)):
        raise ValueError('Exact sixteen calibration identities required')
    indices = []
    for subject in sorted(allowed):
        rows = np.flatnonzero(subjects == subject)
        rows = rows[np.argsort(windows[rows])]
        if not np.array_equal(windows[rows], np.arange(15)):
            raise ValueError('Exactly unique calibration windows 0..14 required')
        indices.extend(rows.tolist())
    return np.array(indices)


def main():
    inner = ROOT/'research/benchmarks/type2branch_inner_train_features_v1'
    prior = json.loads((inner/'plan.json').read_text())
    report = json.loads((inner/'report.json').read_text())
    roles_path = ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    roles = json.loads(roles_path.read_text())
    for group in ['input_sha256', 'source_sha256']:
        for name, expected in prior[group].items():
            if sha(ROOT/name) != expected:
                raise ValueError(f'Frozen preparation dependency changed: {name}')
    calibration = set(roles['roles']['calibration'])
    if (calibration != set(prior['roles']['calibration'])
            or calibration & (set(roles['roles']['fit']) | set(roles['roles']['selection']))):
        raise ValueError('Calibration partition changed')
    array = inner/'calibration.npz'
    if sha(array) != report['roles']['calibration']['sha256']:
        raise ValueError('Calibration artifact changed')
    population = ROOT/'research/benchmarks/type2branch_context_fit_v1/population.npz'
    sources = [Path(__file__), roles_path, inner/'plan.json', inner/'report.json']
    sources += [ROOT/'scripts'/name for name in [
        'type2branch_variable_adapter.py', 'type2branch_keyrecs_adapter.py',
        'type2branch_context_model.py', 'type2branch_csv_bridge.py',
        'type2branch_random.py', 'type2branch_residual_features.py',
        'type2branch_synthesis_cleanup.py', 'typenet_benchmark.py']]
    OUT.mkdir(exist_ok=False)
    plan = {
        'scope': 'Reserved inner TRAIN calibration only; no scores or model selection',
        'lengths': list(LENGTHS), 'identities': sorted(calibration),
        'gallery': 'Windows 0..4, full 100 events, fixed across all lengths',
        'probes': 'Windows 5..14, first L actual raw events, then synthesis and padding',
        'random': 'Separate fresh first-thread stream for gallery and each probe length; sorted identity/window order',
        'threshold_policy': 'After paired checkpoint freeze, one empirical FAR<=1% threshold per exact probe length; no threshold tuning on DEV',
        'limitations': ['Same-session calibration is not cross-session validation',
                       'Correlated length variants are not independent trials',
                       'Actual raw-prefix synthesis differs from feature-prefix training augmentation'],
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [array, population]},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    model = MeanContextModel()
    with np.load(population, allow_pickle=False) as p:
        for f, order, key, count, mean, square in zip(
                p['feature'], p['order'], p['hashes'], p['count'], p['mean'], p['mean_square'], strict=True):
            index = (int(f), int(order), int(key))
            if index in model.models: raise ValueError('Duplicate population entry')
            model.models[index] = (int(count), float(mean), float(square))
    with np.load(array, allow_pickle=False) as data:
        order = ordered_indices(data['subject'], data['window'], calibration)
        raw = data['signed_raw_ms'][order]
        subjects = data['subject'][order]; windows = data['window'][order]
        if raw.shape != (240, 100, 3) or not (data['true_length'] == 100).all():
            raise ValueError('Expected full-length stored calibration events')
    gallery_rng = first_synthesis_thread()
    galleries = {i: synthesize_window(raw_prefix_window(raw[i], 100), model, gallery_rng)
                 for i in range(len(raw)) if windows[i] < 5}
    results = {}
    for length in LENGTHS:
        rng = first_synthesis_thread()
        entries = [galleries[i] if windows[i] < 5 else
                   synthesize_window(raw_prefix_window(raw[i], length), model, rng)
                   for i in range(len(raw))]
        target = OUT/f'length_{length}.npz'
        np.savez_compressed(target, subject=subjects, window=windows,
            features=np.stack([e['features'] for e in entries]),
            residual_valid=np.stack([e['residual_valid'] for e in entries]),
            context_orders=np.stack([e['context_orders'] for e in entries]),
            true_length=np.array([e['true_length'] for e in entries]))
        results[str(length)] = {'sha256': sha(target), 'gallery_n': 80, 'probe_n': 160,
                                'shape': [240, 100, 5]}
    result = {'status': 'raw_prefix_calibration_features_prepared', 'lengths': results,
              'dev_test_payloads_decoded': False, 'metrics': {},
              'limitations': plan['limitations']}
    (OUT/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
