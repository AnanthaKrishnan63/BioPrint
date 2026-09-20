"""Prepare identity-disjoint inner-TRAIN selection/calibration with frozen transforms."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, DATA, sha
from type2branch_context_model import MeanContextModel
from type2branch_keyrecs_adapter import adapt
from type2branch_random import first_synthesis_thread, fill_average_fallback
from type2branch_residual_features import residual_features
from type2branch_synthesis_cleanup import cleanup

OUT = ROOT/'research/benchmarks/type2branch_inner_train_features_v1'


def select_rows(lines, allowed):
    if not next(lines).startswith('participant,session,key1,key2,'):
        raise ValueError('Unexpected header')
    grouped = defaultdict(list)
    for line in lines:
        subject, session, opaque = line.split(',', 2)
        # No key/timing parse until the identity/session guard passes.
        if subject in allowed and session == '1':
            grouped[subject].append(opaque)
    return grouped


def main():
    OUT.mkdir(exist_ok=False)
    roles_path = ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    roles = json.loads(roles_path.read_text())
    for name, expected in roles['source_sha256'].items():
        if sha(ROOT/name) != expected: raise ValueError('Role input changed')
    manifest_path = DATA/'split-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selected = set(roles['roles']['selection'])
    calibrated = set(roles['roles']['calibration'])
    allowed = selected | calibrated
    assert len(selected) == len(calibrated) == 16 and not selected & calibrated
    assert not allowed & set(roles['roles']['fit'])
    assert not allowed & set(manifest['sealed_test_subjects'])
    population_dir = ROOT/'research/benchmarks/type2branch_context_fit_v1'
    population_path = population_dir/'population.npz'
    expected_population = json.loads((population_dir/'report.json').read_text())['population_sha256']
    source = DATA/'free-text.csv'
    plan = {'scope': 'Only32innerTRAINnon-fitting identities/session1 decoded',
            'roles': {'selection': sorted(selected), 'calibration': sorted(calibrated)},
            'windows': 'First15full100event windows per identity, no replacement',
            'scoring': 'First5windows gallery; remaining10probe; identity-disjoint role groups',
            'selection': 'May select checkpoints using innerTRAIN only',
            'calibration': 'Threshold setting only after checkpoint freeze; no fit or selection',
            'population': 'Frozen47fitIDs only; no updates',
            'random': 'Each role gets fresh first-thread stream; sorted subjects/window order',
            'policy': 'Same Average and zero missing-residual adaptation as frozen fit inputs',
            'input_sha256': {str(source.relative_to(ROOT)): manifest['files']['free-text.csv']['sha256'],
                             str(population_path.relative_to(ROOT)): expected_population,
                             str(roles_path.relative_to(ROOT)): sha(roles_path)},
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                [Path(__file__)] + [ROOT/'scripts'/n for n in ['type2branch_keyrecs_adapter.py',
                    'typenet_benchmark.py', 'type2branch_context_model.py', 'type2branch_random.py',
                    'type2branch_residual_features.py', 'type2branch_synthesis_cleanup.py']]}}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT/name) != expected: raise ValueError('Input changed')
    model = MeanContextModel()
    with np.load(population_path, allow_pickle=False) as p:
        for f, order, key, count, mean, square in zip(p['feature'], p['order'], p['hashes'],
                                                    p['count'], p['mean'], p['mean_square'], strict=True):
            identity = (int(f), int(order), int(key))
            assert identity not in model.models
            model.models[identity] = (int(count), float(mean), float(square))
    with source.open() as stream:
        grouped = select_rows(stream, allowed)
    assert set(grouped) == allowed
    results = {}
    for role, identities in [('selection', selected), ('calibration', calibrated)]:
        rng = first_synthesis_thread()
        features, valid, subjects, raw_rows, synthetic, orders_list = [], [], [], [], [], []
        audits = {}
        for subject in sorted(identities):
            raw, base, audit = adapt(grouped[subject])
            if len(base) < 15: raise ValueError('Frozen identity lacks15windows')
            audits[subject] = audit
            for wire, observed in zip(raw[:15], base[:15], strict=True):
                predicted, orders = model.predict(wire[:,0])
                generated = np.column_stack((wire[:,0], fill_average_fallback(predicted, rng)))
                cleaned, partitions = cleanup(generated)
                assert len(partitions) == 0
                x, mask = residual_features(observed, cleaned)
                features.append(x); valid.append(mask); subjects.append(subject)
                raw_rows.append(wire); synthetic.append(cleaned); orders_list.append(orders)
        features, valid, orders_list = map(np.stack, (features, valid, orders_list))
        assert features.shape == (240,100,5) and np.isfinite(features).all()
        path = OUT/f'{role}.npz'
        np.savez_compressed(path, features=features, residual_valid=valid, subject=np.array(subjects),
                            window=np.tile(np.arange(15),16), true_length=np.full(240,100),
                            signed_raw_ms=np.stack(raw_rows), synthetic_ms=np.stack(synthetic),
                            context_orders=orders_list)
        results[role] = {'identities':16, 'shape':list(features.shape), 'sha256':sha(path),
                         'fallback_HT_FT':(orders_list<0).sum(axis=(0,1)).tolist(),
                         'missing_residual_HT_FT':(~valid).sum(axis=(0,1)).tolist(),
                         'per_identity':audits}
    report = {'status':'inner_train_features_prepared', 'roles':results,
              'dev_test_observations_decoded':False,
              'limitations':['Explicit source-order/Average/residual adaptations',
                             'No encoder training or recognition metrics yet']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({role:{k:v for k,v in result.items() if k!='per_identity'} for role,result in results.items()}))


if __name__ == '__main__':
    main()
