"""Prepare the frozen one-capture DEV cohort after completed calibration."""
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, DATA, sha
from type2branch_calibrate_lengths import verify_hashes
from type2branch_capture_calibration_gate import calibrated_capture
from type2branch_capture_policy import allocate_capture
from type2branch_context_model import MeanContextModel
from type2branch_prepare_dev import select_dev_rows
from type2branch_random import first_synthesis_thread
from type2branch_variable_adapter import adapt_variable, synthesize_window

OUT = ROOT/'research/benchmarks/type2branch_capture_dev_features_v1'


def main():
    pair = ROOT/'research/benchmarks/type2branch_length_pair_v1'
    comparison = json.loads((pair/'report.json').read_text())
    arms = {arm: ROOT/f'research/benchmarks/type2branch_length_train_{arm}_v1'
            for arm in ['full', 'mixed']}
    reports = {arm: json.loads((run/'report.json').read_text()) for arm, run in arms.items()}
    calibration = ROOT/'research/benchmarks/type2branch_length_calibration_v1'
    calibrated = json.loads((calibration/'report.json').read_text())
    calibration_plan = json.loads((calibration/'plan.json').read_text())
    thresholds = calibrated_capture(comparison, reports, calibrated, calibration_plan)
    # All completion and calibration prerequisites precede DEV payload access.
    verify_hashes(calibration_plan)
    verify_hashes(json.loads((pair/'plan.json').read_text()))
    for run in arms.values(): verify_hashes(json.loads((run/'plan.json').read_text()))
    checkpoint = Path(calibration_plan['checkpoint']['checkpoint'])
    for name, expected in calibration_plan['checkpoint_sha256'].items():
        if sha(checkpoint.parent/name) != expected: raise ValueError('Calibrated checkpoint changed')
    for length, record in calibrated['lengths'].items():
        if sha(calibration/f'length_{length}_scores.npz') != record['scores_sha256']:
            raise ValueError('Calibration score artifact changed')
    protocol_path = ROOT/'research/benchmarks/type2branch_capture_protocol_v1/protocol.json'
    protocol = json.loads(protocol_path.read_text()); verify_hashes(protocol)
    split = json.loads((DATA/'split-manifest.json').read_text())
    allowed = set(protocol['allowed_identities'])
    if (len(allowed) != 79 or allowed != set(split['active_subjects'])
            or allowed & set(split['sealed_test_subjects'])):
        raise ValueError('Exact active cohort required; sealed identities excluded')
    inner_plan = json.loads((ROOT/'research/benchmarks/type2branch_inner_train_features_v1/plan.json').read_text())
    population = ROOT/'research/benchmarks/type2branch_context_fit_v1/population.npz'
    population_hash = inner_plan['input_sha256'][str(population.relative_to(ROOT))]
    calibration_features_plan = ROOT/'research/benchmarks/type2branch_length_calibration_features_v1/plan.json'
    frozen_population_hash = json.loads(calibration_features_plan.read_text())['input_sha256'][str(population.relative_to(ROOT))]
    if population_hash != frozen_population_hash:
        raise ValueError('DEV population differs from calibrated feature preparation')
    source = DATA/'free-text.csv'
    for path, expected in [(population, population_hash), (source, split['files']['free-text.csv']['sha256'])]:
        if sha(path) != expected: raise ValueError('Frozen dataset/population changed')
    files = [Path(__file__), protocol_path, calibration/'plan.json', calibration/'report.json', pair/'report.json']
    files += [ROOT/'scripts'/name for name in ['type2branch_capture_calibration_gate.py',
        'type2branch_calibrate_lengths.py', 'type2branch_length_completion.py',
        'type2branch_prepare_dev.py', 'type2branch_reference_smoke.py']]
    plan = {'protocol': protocol, 'thresholds': thresholds, 'checkpoint': comparison['selected_checkpoint'],
            'input_sha256': {str(source.relative_to(ROOT)): sha(source), str(population.relative_to(ROOT)): population_hash},
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    model = MeanContextModel()
    with np.load(population, allow_pickle=False) as p:
        for f, order, key, count, mean, square in zip(p['feature'], p['order'], p['hashes'],
                p['count'], p['mean'], p['mean_square'], strict=True):
            index = (int(f), int(order), int(key))
            if index in model.models: raise ValueError('Duplicate population entry')
            model.models[index] = (int(count), float(mean), float(square))
    with source.open() as stream: grouped = select_dev_rows(stream, allowed)
    failures = []; results = {}; coverage = {}
    for role in ['gallery', 'probes']:
        rng = first_synthesis_thread(); entries = []; subjects = []; indices = []; audits = {}
        for subject in sorted(allowed):
            try:
                if role == 'gallery':
                    rows = grouped.get((subject, '1'), [])
                    if len(rows) < 2000: raise ValueError('Required five enrollment windows unavailable')
                    windows, audit = adapt_variable(rows, max_windows=20)
                    chosen = [windows[i] for i in protocol['gallery']['windows']]
                    if len(chosen) != 5 or any(w['true_length'] != 100 for w in chosen):
                        raise ValueError('Required enrollment windows incomplete')
                    audits[subject] = audit
                else:
                    window, audit = allocate_capture(grouped.get((subject, '2'), []))
                    coverage[subject] = audit
                    chosen = [] if window is None else [window]
                for window in chosen:
                    entries.append(synthesize_window(window, model, rng))
                    subjects.append(subject); indices.append(window['index'])
            except (ValueError, IndexError, TypeError) as error:
                failures.append({'role': role, 'subject': subject, 'error': str(error)})
        # Never persist or score a selectively successful subset after failure.
        if failures: continue
        features = np.stack([e['features'] for e in entries]) if entries else np.empty((0, 100, 5))
        if role == 'gallery' and features.shape != (395, 100, 5):
            raise ValueError('Complete gallery shape mismatch')
        if role == 'probes' and len(entries) != sum(v['action'] == 'score' for v in coverage.values()):
            raise ValueError('Probe coverage count mismatch')
        target = OUT/f'{role}.npz'
        np.savez_compressed(target, features=features, subject=np.array(subjects, dtype='U16'),
            window=np.array(indices, dtype=np.int64),
            true_length=np.array([e['true_length'] for e in entries], dtype=np.int64),
            residual_valid=np.stack([e['residual_valid'] for e in entries]) if entries else np.empty((0,100,2),dtype=bool),
            context_orders=np.stack([e['context_orders'] for e in entries]) if entries else np.empty((0,100,2),dtype=np.int16))
        results[role] = {'shape': list(features.shape), 'sha256': sha(target), 'per_identity': audits}
    if not failures and set(coverage) != allowed: raise ValueError('Incomplete coverage accounting')
    report = {'status': 'infeasible_prescribed_cohort' if failures else 'frozen_capture_dev_features_complete',
        'roles': results, 'coverage': coverage, 'failures': failures, 'metrics': {},
        'sealed_test_payloads_decoded': False, 'limitations': protocol['limitations'],
        'prior_exposure': protocol['prior_exposure']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'failures': failures,
        'eligible': sum(v['action'] == 'score' for v in coverage.values()), 'identities': 79}))


if __name__ == '__main__':
    main()
