"""Freeze identity-disjoint inner TRAIN roles using audited prefix counts only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/type2branch_train_roles_v1'
SALT = 'type2branch-inner-train-20260920-v1'


def main():
    OUT.mkdir(exist_ok=False)
    source = ROOT/'research/benchmarks/type2branch_keyrecs_eligibility_v1/report.json'
    split = ROOT/'research/benchmarks/data/keyrecs/split-manifest.json'
    audit, manifest = json.loads(source.read_text()), json.loads(split.read_text())
    active = set(manifest['active_subjects'])
    if set(audit['per_identity']) != active or active & set(manifest['sealed_test_subjects']):
        raise ValueError('Audited identities differ from original TRAIN split')
    ordered = sorted(active, key=lambda s: hashlib.sha256(f'{SALT}:{s}'.encode()).hexdigest())
    first, second = int(len(ordered)*.6), int(len(ordered)*.8)
    roles = {'fit': sorted(ordered[:first]), 'selection': sorted(ordered[first:second]),
             'calibration': sorted(ordered[second:])}
    if min(map(len, roles.values())) < 10:
        raise ValueError('Insufficient identity count for K10')
    if min(audit['per_identity'][s]['full_100_row_windows'] for s in active) < 20:
        raise ValueError('Insufficient15reference sequences plus5support windows')
    assert len(set().union(*(set(v) for v in roles.values()))) == sum(map(len,roles.values()))
    report = {'status': 'inner_train_identity_roles_frozen_by_metadata', 'salt': SALT,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, split, Path(__file__)]},
        'roles': roles, 'role_counts': {k:len(v) for k,v in roles.items()},
        'source_session': '1 only; original DEV session2 and sealed identities unchanged',
        'fit_rule': 'Only fit identities may update global encoder/synthesis population',
        'selection_rule': 'Selection identities may select within TRAIN; never optimizer updates',
        'calibration_rule': 'Calibration identities excluded from encoder fitting and model selection',
        'window_contract_pending': True,
        'limitations': ['Metadata feasibility, not proof of event continuity/quality',
                       'Exact window/support/probe choices must be frozen before decoding observations',
                       'Future DEV reports must distinguish fit and representation-unseen identities',
                       'Small-data protocol adaptation, not published1000-user validation']}
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'roles.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'role_counts':report['role_counts']}))


if __name__ == '__main__':
    main()
