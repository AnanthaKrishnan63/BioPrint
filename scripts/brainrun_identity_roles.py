"""Freeze BrainRun user partitions from identifier metadata only."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/brainrun'
OUT=DATA/'identity_roles_v1'
SEED='bioprint-brainrun-identities-20260920-v1'
CAPS={'fit':120,'selection':40,'calibration':40,'dev':60}


def canonical(value):
    return json.dumps(value,ensure_ascii=True,separators=(',',':'))


def partition(identifier):
    digest=hashlib.sha256((SEED+':'+identifier).encode()).hexdigest()
    bucket=int(digest,16)%100
    role=('fit' if bucket<60 else 'selection' if bucket<70 else
          'calibration' if bucket<80 else 'dev' if bucket<90 else 'test')
    return role,digest


def assign(users):
    assigned={u:partition(u) for u in sorted(set(users))}
    selected={role:set(sorted((u for u,(r,_) in assigned.items() if r==role),
        key=lambda u:assigned[u][1])[:cap]) for role,cap in CAPS.items()}
    result={}
    for user,(role,digest) in assigned.items():
        active=role!='test' and user in selected[role]
        result[user]={'identity_role':role,'partition':'train' if role in ['fit','selection','calibration'] else role,
            'active':active,'behavioral_access':'pending_train_schema_protocol' if active else 'sealed',
            'assignment_sha256':digest}
    return result


def main():
    from brainrun_bson_metadata import iter_metadata
    acquired=DATA/'acquisition_v1';receipt=json.loads((acquired/'report.json').read_text())
    if receipt['status']!='compressed_archive_acquired' or receipt['observations_decoded']:
        raise ValueError('Verified compressed-only acquisition required')
    archive=ROOT/receipt['archive']
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1_048_576),b''):h.update(block)
    if h.hexdigest()!=receipt['sha256']:raise ValueError('Archive checksum mismatch')
    sources=[Path(__file__),ROOT/'scripts/brainrun_bson_metadata.py',acquired/'report.json',DATA/'archive_metadata_v1/report.json']
    plan={'scope':'Only devices BSON identifier fields device_id and user_id; all behavioral/device-context values skipped',
        'seed':SEED,'hash_rule':'SHA256(seed+colon+canonical typed user identifier), integer modulo100',
        'ranges':{'train_fit':[0,59],'train_selection':[60,69],'train_calibration':[70,79],'dev':[80,89],'test':[90,99]},
        'active_caps':CAPS,'active_selection':'Smallest assignment hashes within each non-test role; no behavior/coverage selection',
        'all_other_identities':'sealed, no behavioral decoding','max_document_bytes':16*1024*1024,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'archive_sha256':receipt['sha256']}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    devices={};records=0
    with zipfile.ZipFile(archive) as z:
        with z.open('gestures_devices_users_games_data/devices.bson') as stream:
            for row in iter_metadata(stream,fields=('device_id','user_id')):
                records+=1
                if set(row)!={'device_id','user_id'}:raise ValueError('Published identifier schema mismatch; no inferred repair')
                device,user=canonical(row['device_id']),canonical(row['user_id'])
                if device in devices and devices[device]!=user:raise ValueError('Ambiguous device-to-user mapping')
                devices[device]=user
    if not devices:raise ValueError('No identity metadata')
    users=assign(devices.values())
    if any(not any(v['active'] and v['identity_role']==role for v in users.values()) for role in CAPS):
        raise ValueError('Empty prescribed active role')
    result={'status':'identity_metadata_roles_frozen','device_to_user':devices,'users':users,
        'counts':{'device_records':records,'unique_devices':len(devices),'users':len(users),
            'active_by_role':{r:sum(v['active'] and v['identity_role']==r for v in users.values()) for r in CAPS}},
        'behavioral_observations_decoded':False,'device_context_decoded':False,
        'scope':'Identifier metadata only; no games, gesture, sensor or user-profile members opened'}
    (OUT/'roles.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':main()
