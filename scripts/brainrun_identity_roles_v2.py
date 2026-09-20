"""Freeze stable identity roles after quarantining shared device associations."""
import hashlib
import json
from pathlib import Path

from brainrun_identity_roles import CAPS, SEED, partition
from brainrun_guarded_stream_v2 import _validate_identifier

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'datasets/brainrun/device_mapping_audit_v1'
AUDIT_SHA256 = 'a129b1935343e72a131df3d74b7b91cdc39032df4c93597460fe651c287527d4'
ARCHIVE_SHA256 = '070f43c7a068e96e484d303ad67e85d018859aebbf1c25d6e5b7fe563426bb6e'
OUT = ROOT/'datasets/brainrun/identity_roles_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_quarantined(device_to_users, caps=None):
    """Keep hash assignments stable; apply eligibility before capped selection."""
    caps = CAPS if caps is None else caps
    if set(caps) != set(CAPS) or any(type(n) is not int or n < 1 for n in caps.values()):
        raise ValueError('Positive fit/selection/calibration/dev caps required')
    if not isinstance(device_to_users, dict) or not device_to_users:
        raise ValueError('Nonempty audited mapping required')
    users, shared, affected, unambiguous = set(), [], set(), {}
    for device, owners in device_to_users.items():
        _validate_identifier(device)
        if not isinstance(owners, list) or not owners or len(set(owners)) != len(owners):
            raise ValueError('Distinct nonempty audited owners required')
        for user in owners:_validate_identifier(user)
        users.update(owners)
        if len(owners) > 1:
            shared.append(device); affected.update(owners)
        else:
            unambiguous[device] = owners[0]
    assignments = {user: partition(user) for user in sorted(users)}
    selected = {role: set(sorted((user for user,(r,_) in assignments.items()
                                 if r == role and user not in affected),
                                key=lambda user: assignments[user][1])[:cap])
                for role,cap in caps.items()}
    roles = {}
    for user,(role,checksum) in assignments.items():
        active = role != 'test' and user in selected[role]
        roles[user] = {'identity_role': role,
            'partition': 'train' if role in ['fit','selection','calibration'] else role,
            'active': active, 'assignment_sha256': checksum,
            'behavioral_access': 'pending_train_schema_protocol' if active else 'sealed'}
        if user in affected:roles[user]['inactive_reason'] = 'ambiguous_device_link'
        elif not active:roles[user]['inactive_reason'] = 'test_reserved' if role == 'test' else 'outside_active_cap'
    return {'device_to_user': dict(sorted(unambiguous.items())),
            'quarantine_device_ids': sorted(shared), 'quarantine_user_ids': sorted(affected),
            'users': roles, 'counts': {'devices': len(device_to_users), 'users': len(users),
                'shared_devices': len(shared), 'affected_users': len(affected),
                'active_by_role': {r: len(s) for r,s in selected.items()}}}


def main():
    if digest(AUDIT/'report.json') != AUDIT_SHA256:
        raise ValueError('Complete identifier audit changed')
    audit = json.loads((AUDIT/'report.json').read_text())
    audit_plan = json.loads((AUDIT/'plan.json').read_text())
    if (audit['status'] != 'identifier_mapping_audit_complete'
            or audit['behavioral_observations_decoded'] is not False
            or audit['context_decoded'] is not False or audit_plan['archive_sha256'] != ARCHIVE_SHA256):
        raise ValueError('Verified identifier-only audit required')
    for name,expected in audit_plan['source_sha256'].items():
        if digest(ROOT/name) != expected:raise ValueError('Identifier audit source changed')
    result = assign_quarantined(audit['device_to_users'])
    if any(result['counts'][key] != audit[key] for key in ['devices','users','shared_devices','affected_users']):
        raise ValueError('Audited mapping counts disagree')
    if any(result['counts']['active_by_role'][role] != count for role,count in CAPS.items()):
        raise ValueError('Prescribed active cohort sizes unavailable')
    sources = [Path(__file__), AUDIT/'report.json', AUDIT/'plan.json',
               ROOT/'scripts/brainrun_identity_roles.py', ROOT/'scripts/brainrun_guarded_stream_v2.py']
    plan = {'scope': 'Identifier metadata only; no archive or behavioral observations opened',
            'audit_sha256': AUDIT_SHA256, 'archive_sha256': ARCHIVE_SHA256,
            'seed': SEED, 'active_caps': CAPS,
            'quarantine': 'Every shared device and every user linked to any shared device excluded before active selection',
            'assignment': 'Unchanged v1 hash-role assignment for every user; lowest hashes among unaffected users per role',
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    result.update(status='identity_metadata_roles_frozen', behavioral_observations_decoded=False,
                  device_context_decoded=False, audit_sha256=AUDIT_SHA256)
    (OUT/'roles.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts']),flush=True)


if __name__ == '__main__':main()
