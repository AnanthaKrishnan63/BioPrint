"""Prepare/review then run DEV evaluation-only tap extraction on exact frozen window IDs."""
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
PROPOSAL=ROOT/'research/benchmarks/hmog/tap_dev_evaluation_protocol_v1.json'
PROPOSAL_SHA='fa198593579a92a90d8b120333c8ec14880ec8eccbe5a3858f8953470233995f'
SUPPLEMENT=ROOT/'research/benchmarks/hmog_tap_implementation_contract_v1.md'
FIT=ROOT/'datasets/hmog/dev_features_v1.npz'
FIT_SHA='05f94bd9166d81096d9c667fffcd2bcd9032c1760174b7edd716026c2546b7d5'
PLAN=OUT/'dev_tap_extraction_plan_v1.json'
DEST=ROOT/'datasets/hmog/dev_tap_features_v1.npz'
IDS={'556357','219303','777078','737973'}
COVERAGE=OUT/'validation_v1/validation_complete.json'
OVERLAY=ROOT/'datasets/hmog/tap_dev_evaluation_feature_extraction_authorization_v1.json'
META=('subject','session','role','activity_id','window_index')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def validate_metadata(meta,expected_rows=68):
    if set(meta)!=set(META):raise ValueError('Exact window metadata fields required')
    n=len(meta['subject'])
    if n!=expected_rows or any(v.shape!=(n,) for v in meta.values()):raise ValueError('Frozen window row count/shape mismatch')
    if meta['subject'].dtype.kind not in 'US' or meta['role'].dtype.kind not in 'US':raise ValueError('String identities/roles required')
    if any(meta[n].dtype.kind not in 'iu' for n in ('session','activity_id','window_index')):raise ValueError('Integral session/window metadata required')
    if set(meta['subject'].astype(str))!=IDS:raise PermissionError('All four fixed DEV identities required')
    for s,r,w in zip(meta['session'],meta['role'].astype(str),meta['window_index']):
        if not 1<=s<=16 or r!=('train_support' if s<=8 else 'dev_probe') or w<0:
            raise PermissionError('Unauthorized role/session/window')
    tuples=list(zip(*(meta[n].tolist() for n in ('subject','session','activity_id','window_index'))))
    if len(tuples)!=len(set(tuples)):raise ValueError('Duplicate frozen window identity')
    return meta


def load_metadata():
    if digest(FIT)!=FIT_SHA:raise ValueError('Frozen DEV archive changed')
    with np.load(FIT,allow_pickle=False) as archive:
        meta={name:archive[name].copy() for name in META}
    # key and imu payloads are never accessed.
    return validate_metadata(meta)


def prepare():
    if digest(PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Proposal hash mismatch')
    proposal=json.loads(PROPOSAL.read_text())
    for name in ('dev_window_archive','tap_contract','tap_implementation','role_manifest','fit_parameters','selection_report'):
        binding=proposal['bindings'][name]
        if digest(ROOT/binding['path'])!=binding['sha256']:raise ValueError('Protocol dependency changed: '+name)
    if proposal['window_count']!=68 or proposal['support_windows']!=37 or proposal['probe_windows']!=31:raise ValueError('Protocol window counts changed')
    meta=load_metadata()
    if np.sum(meta['role']=='train_support')!=37 or np.sum(meta['role']=='dev_probe')!=31:raise ValueError('Original role counts changed')
    sources={n:digest(ROOT/'scripts'/n) for n in ['hmog_dev_tap_extract.py','hmog_tap_windows.py','hmog_tap_reference.py','hmog_contact_parser.py','hmog_data.py']}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'proposal_sha256':PROPOSAL_SHA,
        'implementation_contract_path':str(SUPPLEMENT.relative_to(ROOT)),
        'implementation_contract_sha256':digest(SUPPLEMENT),
        'upstream_coverage_path':str(COVERAGE.relative_to(ROOT)),'upstream_coverage_sha256':digest(COVERAGE),
        'guard_overlay_path':str(OVERLAY.relative_to(ROOT)),'guard_overlay_sha256':digest(OVERLAY),
        'dev_window_archive_sha256':FIT_SHA,'sources_sha256':sources,'dev_window_count':68,
        'subjects':sorted(IDS),'phase':'tap_dev_evaluation_feature_extraction','purpose':'tap_dev_evaluation_feature_extraction',
        'raw_members':['Activity.csv','TouchEvent_im.csv'],'metadata_only_archive_fields':list(META),
        'window_rule':'Exact original activity wallstart+30000*window_index; all68originalrows retained',
        'vectors':'Full actual contacts; no observed invalid predecessor gap bridging; pointercount1,portrait,complete,nonoverlap,bothcontacts insidewindow',
        'min_scan_taps':5,'conventions':json.loads((ROOT/'research/benchmarks/hmog_tap_protocol_proposal_v1.json').read_text())['conventions'],'source_snapshot':'dev_tap_extraction_source_v1.py',
        'canonical_paths':{k:str(v.resolve()) for k,v in {'proposal':PROPOSAL,'base_features':FIT,'supplement':SUPPLEMENT,'plan':PLAN,'output':DEST}.items()},
        'enrollment_counting':'Only windows with >=5 taps contribute to downstream80tap enrollmentminimum; allvectorsretained forcoverage, no templatefitting here',
        'feature_output':str(DEST.relative_to(ROOT)),'no_model_fitting':True,'held_out_measurements_read':False,'run_reads_dev_measurements':True,
        'execution_status':'Prepared for root review; does not authorize raw reads',
        'test_measurements_read':False,'prepare_reads_metadata_only':True,
        'scope_caveats':proposal['scope_caveats'],
        'window_metadata':{k:v.tolist() for k,v in meta.items()}}
    with PLAN.open('x') as f:json.dump(plan,f,indent=2)
    with (OUT/plan['source_snapshot']).open('x') as f:f.write(Path(__file__).read_text())
    print(json.dumps({'prepared_plan':str(PLAN.relative_to(ROOT)),'raw_members_read':False,'windows':68}))


def guarded_rows(reader,sid,session,filename):
    with reader.open_member(sid,session,filename,expected_cohort='dev',
        expected_role='train_support' if session<=8 else 'dev_probe',purpose='tap_dev_evaluation_feature_extraction') as m:
        for index,row in enumerate(csv.reader(io.TextIOWrapper(m,encoding='utf-8-sig'))):
            if index>=2000000:raise ValueError('Row bound exceeded')
            if len(row)!=(10 if filename=='Activity.csv' else 11):raise ValueError('Unexpected columns')
            a=[float(v) for v in (row[:9] if filename=='Activity.csv' else row)]
            if not np.isfinite(a).all():raise ValueError('Nonfinite input')
            yield a


def run():
    plan=json.loads(PLAN.read_text())
    current={k:str(v.resolve()) for k,v in {'proposal':PROPOSAL,'base_features':FIT,'supplement':SUPPLEMENT,'plan':PLAN,'output':DEST}.items()}
    if plan.get('canonical_paths')!=current:raise ValueError('Canonical paths changed from reviewedplan')
    if digest(PROPOSAL)!=plan['proposal_sha256'] or digest(FIT)!=plan['dev_window_archive_sha256']:
        raise ValueError('Frozen input changed')
    if digest(SUPPLEMENT)!=plan['implementation_contract_sha256']:raise ValueError('Implementation contract changed')
    for n,sha in plan['sources_sha256'].items():
        if digest(ROOT/'scripts'/n)!=sha:raise ValueError('Reviewed source changed: '+n)
    if digest(COVERAGE)!=plan['upstream_coverage_sha256'] or digest(OVERLAY)!=plan['guard_overlay_sha256']:raise ValueError('Reviewed coverage or guard overlay changed')
    if DEST.exists():raise ValueError('Preserve existing tap archive')
    meta=load_metadata()
    if any(meta[k].tolist()!=plan['window_metadata'][k] for k in META):raise ValueError('Window metadata changed')
    reader=HmogReader(phase=plan['phase'])
    config=Conventions(**plan['conventions']);vectors=[];mapping=[];counts=np.zeros(68,dtype=np.int32)
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
    report={'plan':plan,'original_window_count':68,'scorable_windows':int(np.sum(counts>=5)),
        'tap_vectors':len(vectors),'window_diagnostics':diagnostics,'feature_archive_path':str(DEST.relative_to(ROOT)),
        'feature_archive_sha256':digest(DEST),'feature_archive_bytes':DEST.stat().st_size,
        'held_out_measurements_read':True,'test_measurements_read':False,'model_fitting_performed':False,'raw_rows_exported':False}
    with (OUT/'dev_tap_extraction_results_v1.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'original_windows':68,'scorable':report['scorable_windows'],'vectors':len(vectors)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args();prepare() if args.action=='prepare' else run()
