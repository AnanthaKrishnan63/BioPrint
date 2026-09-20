"""Bounded schema audit of eight hash-selected TRAIN-fit BrainRun users only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from brainrun_guarded_stream_v2 import iter_authorized

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/brainrun'
OUT=ROOT/'research/benchmarks/brainrun_train_schema_v2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    roles_path=DATA/'identity_roles_v3/roles.json';roles=json.loads(roles_path.read_text())
    if roles['status']!='identity_metadata_roles_frozen' or roles['behavioral_observations_decoded']:
        raise ValueError('Frozen identifier-only roles required')
    role_plan=json.loads((roles_path.parent/'plan.json').read_text())
    for name,expected in role_plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Identity assignment source changed')
    fit=sorted((u for u,v in roles['users'].items() if v['active'] and v['identity_role']=='fit'),
        key=lambda u:roles['users'][u]['assignment_sha256'])[:8]
    if len(fit)!=8:raise ValueError('Eight fitting identities required; no substitute cohort')
    receipt_path=DATA/'acquisition_v1/report.json';receipt=json.loads(receipt_path.read_text())
    archive=ROOT/receipt['archive']
    if sha(archive)!=receipt['sha256'] or receipt['sha256']!=role_plan['archive_sha256']:
        raise ValueError('Original archive changed')
    sys.path.insert(0,str(ROOT/'.research-brainrun-deps'))
    import bson
    import pymongo
    if pymongo.version!='4.15.1':raise ValueError('Pinned BSON runtime required')
    sources=[Path(__file__),roles_path,roles_path.parent/'plan.json',receipt_path,
        ROOT/'scripts/brainrun_guarded_stream_v2.py',ROOT/'scripts/brainrun_bson_metadata.py',
        ROOT/'research/benchmarks/brainrun_runtime_install_v1.json',Path(bson.__file__)]
    plan={'scope':'Eight fixed TRAIN-fit identities only; schema diagnostics, no fitting or recognition scores',
        'allowed_users':fit,'max_documents_per_user':{'games':64,'gestures':256},
        'ordering':'First authorized documents in original member order; not a cross-session coverage audit',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'archive_sha256':receipt['sha256'],
        'excluded_roles':['selection','calibration','dev','test','inactive'],'full_extraction':False}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    reports={}
    with zipfile.ZipFile(archive) as z:
        for kind,cap in plan['max_documents_per_user'].items():
            fields={};counts=Counter();nested={}
            with z.open('gestures_devices_users_games_data/'+kind+'.bson') as stream:
                for user,document in iter_authorized(stream,roles['device_to_user'],roles['users'],fit,
                        lambda raw:bson.BSON(raw).decode(),max_per_user=cap,max_selected_records=cap*len(fit),
                        quarantine_device_ids=roles['quarantine_device_ids']):
                    counts[user]+=1
                    for name,value in document.items():
                        fields.setdefault(name,Counter())[type(value).__name__]+=1
                        if isinstance(value,list):
                            info=nested.setdefault(name,{'lengths':Counter(),'element_types':Counter(),'dict_keys':Counter()})
                            info['lengths'][len(value)]+=1
                            for item in value[:3]:
                                info['element_types'][type(item).__name__]+=1
                                if isinstance(item,dict):info['dict_keys'].update(item.keys())
            reports[kind]={'authorized_document_counts':dict(counts),'top_level_field_types':fields,'list_schema':nested,
                'missing_selected_users':[u for u in fit if not counts[u]]}
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Schema source changed during run')
    report={'status':'train_fit_schema_audit_complete','collections':reports,'recognition_metrics':{},
        'selection_calibration_dev_test_observations_decoded':False,'limitations':['Bounded archive-order sample; not session-eligibility evidence']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':{k:v['authorized_document_counts'] for k,v in reports.items()}}),flush=True)


if __name__=='__main__':main()
