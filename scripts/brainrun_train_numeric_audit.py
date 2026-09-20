"""Preserve the fixed TRAIN-fit BSON sample, then audit numeric conventions."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import struct
import sys
import zipfile
from brainrun_guarded_stream_v2 import iter_authorized
from brainrun_bson_metadata import _read_exact

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'research/benchmarks/brainrun_train_schema_v2'
OUT=ROOT/'research/benchmarks/brainrun_train_numeric_v1'
SCHEMA_PLAN_SHA='8d1987557139f596ba44baf7abb8adf5387378762bee6c604b52a5b2b2f5265a'
RAW_BUDGET=10_000_000
POINT_FIELDS=('dx','dy','moveX','moveY','vx','vy','x0','y0')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def finite_number(value):
    return not isinstance(value,bool) and isinstance(value,Real) and math.isfinite(value)


class Numeric:
    def __init__(self):
        self.counts=Counter(dict(total=0,missing=0,invalid_type=0,nonfinite=0,finite=0,integer=0,nonnegative=0,nonnegative_integer=0))
        self.minimum=None;self.maximum=None;self.types=Counter()

    def add(self,value,present=True):
        self.counts['total']+=1
        if not present:self.counts['missing']+=1;return
        self.types[type(value).__name__]+=1
        if isinstance(value,bool) or not isinstance(value,Real):self.counts['invalid_type']+=1;return
        if not math.isfinite(value):self.counts['nonfinite']+=1;return
        self.counts['finite']+=1;self.counts['integer']+=int(value)==value;self.counts['nonnegative']+=value>=0
        self.counts['nonnegative_integer']+=int(value)==value and value>=0
        self.minimum=value if self.minimum is None else min(self.minimum,value)
        self.maximum=value if self.maximum is None else max(self.maximum,value)

    def report(self):
        return {**self.counts,'types':dict(self.types),'minimum':self.minimum,'maximum':self.maximum}


class Profile:
    def __init__(self,kind):
        if kind not in ['games','gestures']:raise ValueError('Unknown collection')
        self.kind=kind;self.counts=Counter();self.by_user=Counter()
        self.numeric={k:Numeric() for k in ['t_start','t_stop','duration']}
        if kind=='games':self.numeric.update({k:Numeric() for k in ['corrects_number','wrongs_number']})
        self.nominal=defaultdict(Counter);self.sessions=defaultdict(set)
        self.points={k:Numeric() for k in POINT_FIELDS};self.unexpected=Counter();self.timestamp_fields=Counter()

    def add(self,user,document):
        self.counts['records']+=1;self.by_user[user]+=1
        if not isinstance(document,dict):self.counts['invalid_document_shape']+=1;return
        for field in self.numeric:
            if field!='duration':self.numeric[field].add(document.get(field),field in document)
        start,stop=document.get('t_start'),document.get('t_stop')
        if finite_number(start) and finite_number(stop):
            self.numeric['duration'].add(stop-start);self.counts['negative_duration']+=stop<start
        else:self.numeric['duration'].add(None,present=False)
        for field in (['game_type','stage'] if self.kind=='games' else ['type','screen']):
            value=document.get(field)
            if (isinstance(value,str) and len(value)<=256) or (type(value) is int):
                self.nominal[field][json.dumps(value,ensure_ascii=True)]+=1
            else:self.counts['invalid_nominal_'+field]+=1
        if self.kind!='gestures':return
        session=document.get('session_id')
        if isinstance(session,str) or type(session) is int:self.sessions[user].add((type(session).__name__,session))
        else:self.counts['invalid_session_id']+=1
        points=document.get('data')
        if not isinstance(points,list):self.counts['invalid_data_shape']+=1;return
        self.counts['empty_data_arrays']+=not points
        for point in points:
            self.counts['points']+=1
            if not isinstance(point,dict):self.counts['invalid_point_shape']+=1;continue
            for field,stats in self.points.items():stats.add(point.get(field),field in point)
            for field in point:
                if not isinstance(field,str):self.counts['invalid_point_field_name']+=1;continue
                if field not in POINT_FIELDS:self.unexpected[field]+=1
                if 'time' in field.lower() or field.lower() in ['t','ts','t_start','t_stop']:
                    self.timestamp_fields[field]+=1

    def report(self):
        return {'counts':dict(self.counts),'authorized_document_counts':dict(self.by_user),
                'numeric':{k:v.report() for k,v in self.numeric.items()},
                'nominal_counts':{k:dict(v) for k,v in self.nominal.items()},
                'session_counts_per_user':{k:len(v) for k,v in self.sessions.items()},
                'point_fields':{k:v.report() for k,v in self.points.items()} if self.kind=='gestures' else {},
                'unexpected_point_fields':dict(self.unexpected),'point_timestamp_fields':dict(self.timestamp_fields)}


def main():
    if sha(SCHEMA/'plan.json')!=SCHEMA_PLAN_SHA:raise ValueError('Original schema plan changed')
    original=json.loads((SCHEMA/'plan.json').read_text())
    prior=json.loads((SCHEMA/'report.json').read_text())
    if prior['status']!='train_fit_schema_audit_complete':raise ValueError('Completed original schema audit required')
    for name,expected in original['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen schema dependency changed')
    fit=original['allowed_users'];caps=original['max_documents_per_user']
    if len(fit)!=8 or caps!={'games':64,'gestures':256}:raise ValueError('Prescribed cohort/caps changed')
    role_path=ROOT/'datasets/brainrun/identity_roles_v3/roles.json'
    roles=json.loads(role_path.read_text())
    receipt_path=ROOT/'datasets/brainrun/acquisition_v1/report.json'
    receipt=json.loads(receipt_path.read_text());archive=ROOT/receipt['archive']
    if sha(archive)!=original['archive_sha256'] or receipt['sha256']!=original['archive_sha256']:
        raise ValueError('Frozen archive changed')
    sys.path.insert(0,str(ROOT/'.research-brainrun-deps'))
    import bson
    import pymongo
    if pymongo.version!='4.15.1':raise ValueError('Pinned BSON decoder required')
    sources=[Path(__file__),SCHEMA/'plan.json',SCHEMA/'report.json',Path(bson.__file__)]
    plan={'scope':'Same eight TRAIN-fit users/caps; preserve raw authorized sample before numeric diagnostics',
          'allowed_users':fit,'max_documents_per_user':caps,'raw_budget_bytes':RAW_BUDGET,
          'archive_sha256':original['archive_sha256'],
          'source_sha256':{**original['source_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in sources}},
          'selection_calibration_dev_test_decode_allowed':False,'feature_or_model_selection':False}
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    users={kind:[] for kind in caps};seen=Counter();raw_bytes=0
    try:
        with zipfile.ZipFile(archive) as zipped:
            for kind,cap in caps.items():
                with (OUT/f'{kind}.bson').open('xb') as target:
                    with zipped.open('gestures_devices_users_games_data/'+kind+'.bson') as stream:
                        for user,raw in iter_authorized(stream,roles['device_to_user'],roles['users'],fit,
                                lambda raw:raw,max_per_user=cap,max_selected_records=cap*len(fit),
                                quarantine_device_ids=roles['quarantine_device_ids']):
                            seen[kind]+=1
                            if raw_bytes+len(raw)>RAW_BUDGET:raise ValueError('Authorized raw sample exceeds10MB budget')
                            target.write(raw);raw_bytes+=len(raw);users[kind].append(user)
                if dict(Counter(users[kind]))!=prior['collections'][kind]['authorized_document_counts']:
                    raise ValueError('Original bounded sample cardinality changed')
        (OUT/'raw_index.json').write_text(json.dumps({'users_in_document_order':users,
            'raw_sha256':{kind:sha(OUT/f'{kind}.bson') for kind in caps}},indent=2)+'\n')
        reports={}
        for kind in caps:
            profile=Profile(kind)
            with (OUT/f'{kind}.bson').open('rb') as stream:
                for user in users[kind]:
                    prefix=_read_exact(stream,4);size=struct.unpack('<i',prefix)[0]
                    raw=prefix+_read_exact(stream,size-4)
                    profile.add(user,bson.BSON(raw).decode())
                if stream.read(1):raise ValueError('Persisted sample/index mismatch')
            reports[kind]=profile.report()
            reports[kind]['missing_selected_users']=[u for u in fit if u not in profile.by_user]
        for name,expected in plan['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Numeric audit source changed')
        result={'status':'train_fit_numeric_audit_complete','collections':reports,'raw_bytes':raw_bytes,
                'raw_sha256':{kind:sha(OUT/f'{kind}.bson') for kind in caps},
                'selection_calibration_dev_test_observations_decoded':False,'recognition_metrics':{},
                'limitations':['Same bounded archive-order sample; numeric summaries are not session-eligibility or accuracy evidence',
                               'Timestamp units are not inferred from magnitudes; no point timestamps manufactured']}
        (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'raw_bytes':raw_bytes}),flush=True)
    except Exception as error:
        failure={'status':'train_fit_numeric_audit_failed','error_type':type(error).__name__,
                 'authorized_records_seen':dict(seen),'authorized_records_persisted':{k:len(v) for k,v in users.items()},
                 'raw_bytes':raw_bytes,'selection_calibration_dev_test_observations_decoded':False}
        (OUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise


if __name__=='__main__':main()
