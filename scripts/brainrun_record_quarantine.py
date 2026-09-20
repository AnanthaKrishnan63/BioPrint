"""Extend device quarantine from complete identifier audits; preserve all roles."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ROLES=ROOT/'datasets/brainrun/identity_roles_v2'
AUDIT=ROOT/'research/benchmarks/brainrun_record_identity_audit_v1'
OUT=ROOT/'datasets/brainrun/identity_roles_v3'


def extend(roles,audit):
    if audit['status']!='record_identifier_metadata_audit_complete' or audit['behavioral_observations_decoded'] is not False:
        raise ValueError('Complete identifier-only audit required')
    if set(audit['collections'])!={'games','gestures'}:
        raise ValueError('Both collection audits required')
    unknown=set();counts={}
    for collection,record in audit['collections'].items():
        c=record['counts'];entries=record['unknown_devices']
        if (not record['member_crc_verified_at_eof'] or c['missing_device'] or c['conflicting_user']
                or c['records']!=sum(c[k] for k in ['known','quarantine','unknown','missing_device'])
                or c['unknown']!=sum(v['records'] for v in entries.values())
                or record['unknown_device_count']!=len(entries)):
            raise ValueError('Incomplete or conflicting identity metadata')
        unknown.update(entries);counts[collection]=copy.deepcopy(c)
    if unknown & set(roles['device_to_user']):raise ValueError('Unknown devices overlap known mapping')
    result=copy.deepcopy(roles)
    result['quarantine_device_ids']=sorted(set(roles['quarantine_device_ids'])|unknown)
    result['orphan_device_ids']=sorted(unknown)
    result['record_identity_coverage']=counts
    result['quarantine_policy']='Shared devices/users remain excluded; additionally skip only audited unmapped device IDs. Never infer owners. Unknown future devices still fail.'
    result['identity_cohort_changed']=False
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    roles_path=ROLES/'roles.json';audit_path=AUDIT/'report.json'
    roles=json.loads(roles_path.read_text());audit=json.loads(audit_path.read_text())
    if audit['roles_sha256']!=sha(roles_path):raise ValueError('Audit/role provenance mismatch')
    for directory in [ROLES,AUDIT]:
        plan=json.loads((directory/'plan.json').read_text())
        for name,expected in plan['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Recorded source changed')
    result=extend(roles,audit)
    sources=[Path(__file__),roles_path,ROLES/'plan.json',audit_path,AUDIT/'plan.json',ROOT/'scripts/brainrun_guarded_stream_v2.py']
    plan={'scope':'Metadata-only explicit record quarantine, no observation decoding or participant reselection',
        'archive_sha256':audit['archive_sha256'],'parent_roles_sha256':sha(roles_path),'record_audit_sha256':sha(audit_path),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'roles.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'active_counts':result['counts']['active_by_role'],'orphan_devices':len(result['orphan_device_ids']),
        'quarantined_devices':len(result['quarantine_device_ids']),'identity_cohort_changed':False}),flush=True)


if __name__=='__main__':main()
