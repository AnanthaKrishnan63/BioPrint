"""Prepare/review then run FIT-only tap extraction on exact frozen window IDs."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from hmog_data import HmogReader
from hmog_tap_windows import tap_window
from hmog_tap_reference import Conventions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/hmog'
PROPOSAL=ROOT/'research/benchmarks/hmog_tap_protocol_proposal_v1.json'
PROPOSAL_SHA='4ca50d1115d18f1fe0d49bee94fc266648cf360c9862d837a995ac688902f6d3'
SUPPLEMENT=ROOT/'research/benchmarks/hmog_tap_implementation_contract_v1.md'
FIT=ROOT/'datasets/hmog/fit_features_v3.npz'
FIT_SHA='2afe1a18164334e407beaf01dcaadc7613584fbc85d44fcdf29d0a8854cbdb36'
PLAN=OUT/'fit_tap_extraction_plan_v1.json'
DEST=ROOT/'datasets/hmog/fit_tap_features_v1.npz'
IDS={'717868','526319','986737','539502'}
META=('subject','session','role','activity_id','window_index')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def validate_metadata(meta,expected_rows=207):
    if set(meta)!=set(META):raise ValueError('Exact window metadata fields required')
    n=len(meta['subject'])
    if n!=expected_rows or any(v.shape!=(n,) for v in meta.values()):raise ValueError('Frozen window row count/shape mismatch')
    if meta['subject'].dtype.kind not in 'US' or meta['role'].dtype.kind not in 'US':raise ValueError('String identities/roles required')
    if any(meta[n].dtype.kind not in 'iu' for n in ('session','activity_id','window_index')):raise ValueError('Integral session/window metadata required')
    if not set(meta['subject'].astype(str)).issubset(IDS):raise PermissionError('Only fixed FIT identities allowed')
    for s,r,w in zip(meta['session'],meta['role'].astype(str),meta['window_index']):
        if not 1<=s<=16 or r!=('train_enrollment' if s<=8 else 'train_fit') or w<0:
            raise PermissionError('Unauthorized role/session/window')
    tuples=list(zip(*(meta[n].tolist() for n in ('subject','session','activity_id','window_index'))))
    if len(tuples)!=len(set(tuples)):raise ValueError('Duplicate frozen window identity')
    return meta


def load_metadata():
    if digest(FIT)!=FIT_SHA:raise ValueError('Frozen FIT archive changed')
    with np.load(FIT,allow_pickle=False) as archive:
        meta={name:archive[name].copy() for name in META}
    # key and imu payloads are never accessed.
    return validate_metadata(meta)


def prepare():
    if digest(PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Proposal hash mismatch')
    proposal=json.loads(PROPOSAL.read_text())
    for relative,sha in proposal['source_sha256'].items():
        if digest(ROOT/relative)!=sha:raise ValueError('Proposal dependency changed: '+relative)
    meta=load_metadata()
    sources={n:digest(ROOT/'scripts'/n) for n in ['hmog_fit_tap_extract.py','hmog_tap_windows.py','hmog_tap_reference.py','hmog_contact_parser.py','hmog_data.py']}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'proposal_sha256':PROPOSAL_SHA,
        'implementation_contract_path':str(SUPPLEMENT.relative_to(ROOT)),
        'implementation_contract_sha256':digest(SUPPLEMENT),
        'fit_window_archive_sha256':FIT_SHA,'sources_sha256':sources,'fit_window_count':207,
        'subjects':sorted(IDS),'phase':'tap_reference_feature_extraction','purpose':'tap_reference_feature_extraction',
        'raw_members':['Activity.csv','TouchEvent_im.csv'],'metadata_only_archive_fields':list(META),
        'window_rule':'Exact original activity wallstart+30000*window_index; all207originalrows retained',
        'vectors':'Full actual contacts; no observed invalid predecessor gap bridging; pointercount1,portrait,complete,nonoverlap,bothcontacts insidewindow',
        'min_scan_taps':5,'conventions':proposal['conventions'],'source_snapshot':'fit_tap_extraction_source_v1.py',
        'enrollment_counting':'Only windows with >=5 taps contribute to downstream80tap enrollmentminimum; allvectorsretained forcoverage, no templatefitting here',
        'feature_output':str(DEST.relative_to(ROOT)),'no_model_fitting':True,'held_out_measurements_read':False,
        'execution_status':'Prepared for root review; does not authorize raw reads',
        'window_metadata':{k:v.tolist() for k,v in meta.items()}}
    with PLAN.open('x') as f:json.dump(plan,f,indent=2)
    with (OUT/plan['source_snapshot']).open('x') as f:f.write(Path(__file__).read_text())
    print(json.dumps({'prepared_plan':str(PLAN.relative_to(ROOT)),'raw_members_read':False,'windows':207}))


def guarded_rows(reader,sid,session,filename):
    with reader.open_member(sid,session,filename,expected_cohort='fit',
        expected_role='train_enrollment' if session<=8 else 'train_fit',purpose='tap_reference_feature_extraction') as m:
        for index,row in enumerate(csv.reader(io.TextIOWrapper(m,encoding='utf-8-sig'))):
            if index>=2000000:raise ValueError('Row bound exceeded')
            if len(row)!=(10 if filename=='Activity.csv' else 11):raise ValueError('Unexpected columns')
            a=[float(v) for v in (row[:9] if filename=='Activity.csv' else row)]
            if not np.isfinite(a).all():raise ValueError('Nonfinite input')
            yield a


def run():
    plan=json.loads(PLAN.read_text())
    if digest(PROPOSAL)!=plan['proposal_sha256'] or digest(FIT)!=plan['fit_window_archive_sha256']:
        raise ValueError('Frozen input changed')
    if digest(SUPPLEMENT)!=plan['implementation_contract_sha256']:raise ValueError('Implementation contract changed')
    for n,sha in plan['sources_sha256'].items():
        if digest(ROOT/'scripts'/n)!=sha:raise ValueError('Reviewed source changed: '+n)
    if DEST.exists():raise ValueError('Preserve existing tap archive')
    meta=load_metadata()
    if any(meta[k].tolist()!=plan['window_metadata'][k] for k in META):raise ValueError('Window metadata changed')
    reader=HmogReader(phase=plan['phase'])
    config=Conventions(**plan['conventions']);vectors=[];mapping=[];counts=np.zeros(207,dtype=np.int32)
    diagnostics={};groups={}
    for row,(sid,s) in enumerate(zip(meta['subject'].astype(str),meta['session'])):groups.setdefault((sid,int(s)),[]).append(row)
    for (sid,session),indices in groups.items():
        activities={int(a[0]):a for a in guarded_rows(reader,sid,session,'Activity.csv')}
        touch=list(guarded_rows(reader,sid,session,'TouchEvent_im.csv'))
        for row in indices:
            aid=int(meta['activity_id'][row]);wi=int(meta['window_index'][row])
            if aid not in activities:raise ValueError('Original activity missing')
            activity=activities[aid]
            if activity[1]!=int(sid) or activity[2]!=session:raise ValueError('Activity identity mismatch')
            left=activity[3]+30000*wi;right=left+30000
            if right>activity[4]:raise ValueError('Original fullwindow exceeds Activity bounds')
            stream=[r for r in touch if r[2]==aid]
            result=tap_window(stream,session=f'{sid}:{session}',activity=aid,start_ms=left,end_ms=right,conventions=config)
            matrix=result.pop('tap_vectors');result.pop('scan_mean')
            count=len(matrix);counts[row]=count
            vectors.extend(matrix);mapping.extend([row]*count)
            result['observed_down_events_in_window']=sum(left<=r[0]<right and r[5] in (0,5) for r in stream)
            diagnostics[str(row)]=result
    arrays={'tap_vectors':np.asarray(vectors,dtype=np.float64).reshape(-1,11),
            'tap_window':np.asarray(mapping,dtype=np.int32),'window_tap_count':counts,'window_scorable':counts>=5,
            **{'window_'+n:meta[n] for n in META}}
    with DEST.open('xb') as f:np.savez_compressed(f,**arrays)
    report={'plan':plan,'original_window_count':207,'scorable_windows':int(np.sum(counts>=5)),
        'tap_vectors':len(vectors),'window_diagnostics':diagnostics,'feature_archive_path':str(DEST.relative_to(ROOT)),
        'feature_archive_sha256':digest(DEST),'feature_archive_bytes':DEST.stat().st_size,
        'held_out_measurements_read':False,'model_fitting_performed':False,'raw_rows_exported':False}
    with (OUT/'fit_tap_extraction_results_v1.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'original_windows':207,'scorable':report['scorable_windows'],'vectors':len(vectors)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args();prepare() if args.action=='prepare' else run()
