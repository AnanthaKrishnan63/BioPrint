"""Selection-cohort TRAIN feature extraction using frozen fit-v3 operators."""
import csv
from collections import Counter, defaultdict
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from hmog_data import HmogReader,MANIFEST_SHA256
from hmog_clock_audit import complete_key_pairs, summary
from hmog_contact_parser import parse_contacts
from hmog_behaveformer_features import key_features,imu_window_features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/hmog'
DEST=ROOT/'datasets/hmog/selection_features_v1.npz'
SUBJECTS=['180679','622852']
WRITING={3,4,9,10,15,16,21,22}


from hmog_fit_feature_extract_v3 import strict_pairs,window_quality,gap_statistics


def session_inventory(entries, subjects):
    selected={};excluded={}
    for sid in subjects:
        selected[sid]=[];excluded[sid]={}
        for session in range(1,17):
            name=f'{sid}/{sid}_session_{session}/KeyPressEvent.csv'
            entry=entries.get(name)
            if entry is None or entry['uncompressed_bytes']==0:
                excluded[sid][str(session)]={'reason':'missing_keypress' if entry is None else 'empty_keypress',
                    'basis':'archive_metadata','activity_contents_inspected':False,'candidate_windows':None,'eligible_windows':None}
            else:selected[sid].append(session)
    return selected,excluded


def read_rows(reader,sid,session,filename,columns):
    with reader.open_member(sid,session,filename,expected_cohort='selection',
        expected_role='train_enrollment' if session<=8 else 'train_selection',purpose='selection_feature_extraction_v2') as member:
        for n,row in enumerate(csv.reader(io.TextIOWrapper(member,encoding='utf-8-sig'))):
            if n>=2000000 or len(row)!=columns:raise ValueError('Row/column bound')
            vals=[float(v) for v in (row[:9] if filename=='Activity.csv' else row)]
            if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite source')
            yield vals


def main():
    reader=HmogReader(phase='selection_v2_feature_extraction')
    sessions,excluded_sessions=session_inventory(reader.entries,SUBJECTS)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    fit_plan=json.loads((OUT/'fit_feature_v3_plan.json').read_text())
    if digest(ROOT/'scripts/hmog_fit_feature_extract_v3.py')!=fit_plan['source_sha256']:
        raise ValueError('Frozen fit extractor changed')
    for name in ['hmog_clock_audit.py','hmog_behaveformer_features.py','hmog_contact_parser.py']:
        if digest(ROOT/'scripts'/name)!=fit_plan['dependency_sha256'][name]:
            raise ValueError('Frozen pure feature dependency changed')
    if digest(OUT/'model_protocol_v2.json')!=fit_plan['model_protocol_sha256']:
        raise ValueError('Model protocol changed')
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'subject_sessions':sessions,'excluded_sessions':excluded_sessions,
        'role_manifest_sha256':MANIFEST_SHA256,'phase':'selection_v2_feature_extraction','purpose':'selection_feature_extraction_v2',
        'source_sha256':digest(Path(__file__)),
        'dependency_sha256':{n:digest(ROOT/'scripts'/n) for n in ['hmog_data.py','hmog_clock_audit.py','hmog_behaveformer_features.py','hmog_contact_parser.py','hmog_fit_feature_extract_v3.py']},
        'rules':{'window':'30s halfopen Activity wallstart aligned fullwindows; writing tasks andportrait',
          'writing_tasks':sorted(WRITING),'keys':'First52 actualcomplete uniquelymatched perpointercontacts; gap-filtered retainedpair intervals explicitly allowed; no endpoint synthesis; native numericKeyID',
          'touch':'Pure perpointerparser; actions0/5down1/6up,CANCELclearsactivityactivecontacts; portrait/positiveintegralcount/uniqueevents required throughoutusedcontact; unrelatedcontacts do notinvalidatewindow',
          'sensor':'Allwindow rows portrait. Require finite native clocks and activity-file native+SysTime nondecreasing. Nevercompare ns/1e6 to Activityrelative origin. All100bins eachsensor nonempty,N>=3.',
          'features':'Pure sourceformula fixedscales; windowlocalFFT/gradient beforebinning; key50x10 imu100x24 float32',
          'clock':'Recording-time association only,no correction,sharedorigin claimorretiming',
          'quality':'Modelprotocolv2 retainedpair adaptation; no additional gapratio threshold; allchanges frozen before reads'},
        'tap_features':'Deferred to preserve core strictcontract; no synthesizedtap values',
        'model_protocol_path':'research/benchmarks/hmog/model_protocol_v2.json',
        'model_protocol_sha256':digest(OUT/'model_protocol_v2.json'),
        'destination':str(DEST.relative_to(ROOT)),'no_model_or_heldout_reads':True}
    if DEST.exists():raise ValueError('Preserve existing feature archive')
    with (OUT/'selection_feature_v1_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    with (OUT/'selection_feature_v1_source.py').open('x') as f:f.write(Path(__file__).read_text())
    kk=[];ii=[];metadata=[];reports={};gaps=[]
    for sid,ss in sessions.items():
        reports[sid]={}
        for session in ss:
            data={n:list(read_rows(reader,sid,session,n,c)) for n,c in [('Activity.csv',10),
                ('KeyPressEvent.csv',6),('TouchEvent_im.csv',11),('Accelerometer.csv',7),('Gyroscope.csv',7)]}
            keys=data['KeyPressEvent.csv'];touch=data['TouchEvent_im.csv'];pairs=strict_pairs(keys,touch)
            counter=Counter();candidate=0;eligible=0;session_gaps=[]
            for a in data['Activity.csv']:
                if a[8] not in WRITING:continue
                for index in range(int((a[4]-a[3])//30000)):
                    candidate+=1;left=a[3]+index*30000;right=left+30000
                    reasons,pp,sensors=window_quality(keys,touch,pairs,data['Accelerometer.csv'],data['Gyroscope.csv'],left,right,a)
                    if reasons:counter.update(reasons);continue
                    k=key_features([p[0] for p in pp],[p[1] for p in pp],[p[3] for p in pp])
                    ac,gy=sensors
                    im=imu_window_features([r[0] for r in ac],[r[3:6] for r in ac],
                        [r[0] for r in gy],[r[3:6] for r in gy],start_ms=left,end_ms=right)
                    k=k.astype(np.float32);im=im.astype(np.float32)
                    if not np.isfinite(k).all() or not np.isfinite(im).all():raise ValueError('float32 nonfinite')
                    gap=gap_statistics(keys,pp);gaps.append(gap);session_gaps.append(gap)
                    kk.append(k);ii.append(im);metadata.append((sid,session,'train_enrollment' if session<=8 else 'train_selection',int(a[0]),index))
                    eligible+=1
            reports[sid][str(session)]={'candidate_windows':candidate,'eligible_windows':eligible,
                'excluded_windows':candidate-eligible,'exclusion_reasons_nonexclusive':dict(counter),
                'eligible_window_gap_statistics':session_gaps}
    arrays={'key':np.asarray(kk,dtype=np.float32).reshape(-1,50,10),
            'imu':np.asarray(ii,dtype=np.float32).reshape(-1,100,24),
            'subject':np.asarray([m[0] for m in metadata],dtype='U6'),
            'session':np.asarray([m[1] for m in metadata],dtype=np.int16),
            'role':np.asarray([m[2] for m in metadata],dtype='U16'),
            'activity_id':np.asarray([m[3] for m in metadata],dtype=np.int64),
            'window_index':np.asarray([m[4] for m in metadata],dtype=np.int16)}
    with DEST.open('xb') as f:np.savez_compressed(f,**arrays)
    result={'plan':plan,'subjects':reports,'excluded_sessions':excluded_sessions,'total_eligible_windows':len(kk),
            'scope':'two selection identities TRAIN only; no fitting or dev/test reads','version':1,
            'phase':'selection_v2_feature_extraction','held_out_measurements_read':False,
            'feature_archive_path':str(DEST.relative_to(ROOT)),
            'feature_archive_sha256':digest(DEST),
            'feature_file_sha256':digest(DEST),'feature_file_bytes':DEST.stat().st_size,
            'gap_summary':{'retention_ratio':summary([g['retained_endpoint_ratio'] for g in gaps]),
                'extra_events':summary([g['extra_recorded_events'] for g in gaps]),
                'orphan_events':summary([g['orphan_or_ambiguous_extra_events'] for g in gaps]),
                'gap_extra_event_counts':summary([n for g in gaps for n in g['gap_extra_event_counts']]),
                'retained_press_intervals_ms':summary([n for g in gaps for n in g['retained_press_intervals_ms']])},
            'shapes':{k:list(v.shape) for k,v in arrays.items()},'no_model_or_heldout_reads':True}
    with (OUT/'selection_feature_v1_results.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'eligible':len(kk),'shapes':result['shapes'],'bytes':result['feature_file_bytes']}))

if __name__=='__main__':main()
