"""Strict v2 fit-only feature extraction; no model fitting or held-out reads."""
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
from hmog_clock_audit import complete_key_pairs
from hmog_behaveformer_features import key_features,imu_window_features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/hmog'
DEST=ROOT/'datasets/hmog/fit_features_v2.npz'
SUBJECTS=['717868','526319','986737','539502']
WRITING={3,4,9,10,15,16,21,22}


def read_rows(reader,sid,session,filename,columns):
    with reader.open_member(sid,session,filename,expected_cohort='fit',
        expected_role='train_enrollment' if session<=8 else 'train_fit',purpose='feature_extraction') as member:
        for n,row in enumerate(csv.reader(io.TextIOWrapper(member,encoding='utf-8-sig'))):
            if n>=2000000 or len(row)!=columns:raise ValueError('Row/column bound')
            vals=[float(v) for v in (row[:9] if filename=='Activity.csv' else row)]
            if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite source')
            yield vals


def strict_pairs(keys,touch):
    pairs,_=complete_key_pairs(keys)
    events=defaultdict(list)
    contact_rows=[]
    for r in touch:
        if r[5] in (0,1) and r[3]==1 and r[10]==0:
            events[(r[2],r[1],r[5])].append(r)
            contact_rows.append([r[0],r[1],r[2],r[5],r[4],r[10]])
    contacts,_=complete_key_pairs(contact_rows)
    valid=set(contacts)
    accepted=[]
    for p,u,a,k,o in pairs:
        ds,us=events[(a,p,0)],events[(a,u,1)]
        if o!=0 or len(ds)!=1 or len(us)!=1:continue
        d,v=ds[0],us[0]
        if d[4]!=v[4] or (p,u,a,d[4],0) not in valid or v[0]<d[0]:continue
        accepted.append((p,u,a,k,d[0],v[0]))
    return sorted(accepted)


def no_intervening_bad_keys(keys,pairs):
    """Do not bridge an omitted/duplicate/intervening event in retained52 span."""
    if len(pairs)!=52:return False
    lo=min(p[0] for p in pairs);hi=max(p[1] for p in pairs);aid=pairs[0][2]
    observed=Counter(tuple(r[1:]) for r in keys if r[2]==aid and lo<=r[1]<=hi)
    expected=Counter()
    for p,u,a,k,_,_ in pairs:
        expected[(p,a,0,k,0)]+=1;expected[(u,a,1,k,0)]+=1
    return observed==expected


def window_quality(keys,touch,pairs,acc,gyro,start,end,activity):
    reasons=[]
    tt=[r for r in touch if r[2]==activity[0] and start<=r[0]<end]
    if any(r[5] not in (0,1,2) for r in tt):reasons.append('cancel_outside_or_multitouch_action')
    if any(r[3]!=1 for r in tt):reasons.append('touch_not_single_pointer')
    if any(r[10]!=0 for r in tt):reasons.append('touch_nonportrait')
    # Include native-time contact interiors to prevent delayed cancellation escaping the wall window.
    pp=[p for p in pairs if p[2]==activity[0] and start<=p[4]<end and start<=p[5]<end
        and activity[5]<=p[0]<=p[1]<=activity[6]][:52]
    if len(pp)<52:reasons.append('fewer_than_52_unique_matched_pairs')
    else:
        if not no_intervening_bad_keys(keys,pp):reasons.append('invalid_intervening_key_event')
        lo=min(p[0] for p in pp);hi=max(p[1] for p in pp)
        interior=[r for r in touch if r[2]==activity[0] and lo<=r[1]<=hi]
        if any(r[5] not in (0,1,2) or r[3]!=1 or r[10]!=0 for r in interior):
            reasons.append('invalid_native_contact_span')
        anchors=sorted([(p[0],p[4]) for p in pp]+[(p[1],p[5]) for p in pp])
        if any(b[1]<a[1] for a,b in zip(anchors,anchors[1:])):reasons.append('backward_touch_anchor')
    sensors=[]
    for label,source in [('acc',acc),('gyro',gyro)]:
        ar=[r for r in source if r[2]==activity[0]]
        if any(b[1]<a[1] for a,b in zip(ar,ar[1:])):reasons.append(label+'_native_clock_backstep')
        if any(b[0]<a[0] for a,b in zip(ar,ar[1:])):reasons.append(label+'_recording_clock_backstep')
        win=[r for r in ar if start<=r[0]<end]
        if any(r[6]!=0 for r in win):reasons.append(label+'_nonportrait')
        if len(win)<3:reasons.append(label+'_fewer_than_3_records')
        counts=Counter(min(99,int((r[0]-start)//300)) for r in win)
        if len(counts)!=100:reasons.append(label+'_empty_bins')
        sensors.append(win)
    return reasons,pp,sensors


def main():
    reader=HmogReader(phase='fit_feature_extraction')
    sessions={sid:sorted(e['session'] for e in reader.entries.values() if e['subject']==sid
        and e['session']<=16 and e['name'].endswith('/KeyPressEvent.csv') and e['uncompressed_bytes']>0)
        for sid in SUBJECTS}
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'subject_sessions':sessions,
        'role_manifest_sha256':MANIFEST_SHA256,'phase':'fit_feature_extraction','purpose':'feature_extraction',
        'source_sha256':digest(Path(__file__)),
        'dependency_sha256':{n:digest(ROOT/'scripts'/n) for n in ['hmog_data.py','hmog_clock_audit.py','hmog_behaveformer_features.py']},
        'rules':{'window':'30s halfopen Activity wallstart aligned fullwindows; writing tasks andportrait',
          'writing_tasks':sorted(WRITING),'keys':'First52 complete uniquelymatched samepointercontacts, relative holds, numeric recordedKeyID; exactly104 keyevents inretainedspan else rejectwholewindow',
          'touch':'Rejectwindow if CANCEL/OUTSIDE/multitouch action, pointercount!=1 or nonportrait withinrecordingwindow orretained nativecontactspan',
          'sensor':'Allwindow rows portrait. Require finite native clocks and activity-file native+SysTime nondecreasing. Nevercompare ns/1e6 to Activityrelative origin. All100bins eachsensor nonempty,N>=3.',
          'features':'Pure sourceformula fixedscales; windowlocalFFT/gradient beforebinning; key50x10 imu100x24 float32',
          'clock':'Recording-time association only,no correction,sharedorigin claimorretiming',
          'quality':'No criteria relaxation; zero eligible windows isreportable'},
        'tap_features':'Deferred to preserve core strictcontract; no synthesizedtap values',
        'destination':str(DEST.relative_to(ROOT)),'no_model_or_heldout_reads':True}
    if DEST.exists():raise ValueError('Preserve existing feature archive')
    with (OUT/'fit_feature_v2_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    with (OUT/'fit_feature_v2_source.py').open('x') as f:f.write(Path(__file__).read_text())
    kk=[];ii=[];metadata=[];reports={}
    for sid,ss in sessions.items():
        reports[sid]={}
        for session in ss:
            data={n:list(read_rows(reader,sid,session,n,c)) for n,c in [('Activity.csv',10),
                ('KeyPressEvent.csv',6),('TouchEvent_im.csv',11),('Accelerometer.csv',7),('Gyroscope.csv',7)]}
            keys=data['KeyPressEvent.csv'];touch=data['TouchEvent_im.csv'];pairs=strict_pairs(keys,touch)
            counter=Counter();candidate=0;eligible=0
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
                    kk.append(k);ii.append(im);metadata.append((sid,session,'train_enrollment' if session<=8 else 'train_fit',int(a[0]),index))
                    eligible+=1
            reports[sid][str(session)]={'candidate_windows':candidate,'eligible_windows':eligible,
                'excluded_windows':candidate-eligible,'exclusion_reasons_nonexclusive':dict(counter)}
    arrays={'key':np.asarray(kk,dtype=np.float32).reshape(-1,50,10),
            'imu':np.asarray(ii,dtype=np.float32).reshape(-1,100,24),
            'subject':np.asarray([m[0] for m in metadata],dtype='U6'),
            'session':np.asarray([m[1] for m in metadata],dtype=np.int16),
            'role':np.asarray([m[2] for m in metadata],dtype='U16'),
            'activity_id':np.asarray([m[3] for m in metadata],dtype=np.int64),
            'window_index':np.asarray([m[4] for m in metadata],dtype=np.int16)}
    with DEST.open('xb') as f:np.savez_compressed(f,**arrays)
    result={'plan':plan,'subjects':reports,'total_eligible_windows':len(kk),
            'scope':'four fit identities TRAIN only; no population fitting','version':2,
            'phase':'fit_feature_extraction','held_out_measurements_read':False,
            'feature_archive_path':str(DEST.relative_to(ROOT)),
            'feature_archive_sha256':digest(DEST),
            'feature_file_sha256':digest(DEST),'feature_file_bytes':DEST.stat().st_size,
            'shapes':{k:list(v.shape) for k,v in arrays.items()},'no_model_or_heldout_reads':True}
    with (OUT/'fit_feature_v2_results.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'eligible':len(kk),'shapes':result['shapes'],'bytes':result['feature_file_bytes']}))

if __name__=='__main__':main()
