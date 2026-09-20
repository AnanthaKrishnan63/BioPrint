"""Fit-only recording-time paired-window feasibility; counts, no features/models."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from hmog_data import HmogReader, MANIFEST_SHA256
from hmog_clock_audit import complete_key_pairs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/hmog'
SUBJECTS=['717868','526319','986737','539502']
WRITING={3,4,9,10,15,16,21,22}
WIDTH=30000


def bins_complete(times, start):
    """Half-open bins: a right boundary belongs only to the next bin/window."""
    counts=[0]*100
    for t in times:
        if start <= t < start+WIDTH:
            counts[min(99,int((t-start)//300))]+=1
    return {'records':sum(counts),'nonempty_bins':sum(c>0 for c in counts),
            'complete':sum(counts)>=3 and all(counts)}


def matched_pairs(keys,touches):
    pairs,rejections=complete_key_pairs(keys)
    events=defaultdict(list)
    contact_rows=[]
    for r in touches:
        if r[5] not in (0,1,5,6) or r[10]!=0:
            continue
        action=0 if r[5] in (0,5) else 1
        events[(r[2],r[1],action)].append(r)
        contact_rows.append([r[0],r[1],r[2],action,r[4],r[10]])
    contacts,_=complete_key_pairs(contact_rows)
    valid_contacts={(p,r,a,pointer,o) for p,r,a,pointer,o in contacts}
    matched=[]
    rejected=Counter()
    for press,release,activity,key,orientation in pairs:
        if orientation!=0:
            rejected['nonportrait_pair']+=1
            continue
        down,up=events[(activity,press,0)],events[(activity,release,1)]
        if len(down)!=1 or len(up)!=1:
            rejected['missing_or_nonunique_touch_anchor']+=1
            continue
        d,u=down[0],up[0]
        if d[4]!=u[4] or (press,release,activity,d[4],0) not in valid_contacts:
            rejected['not_same_complete_touch_contact']+=1
            continue
        if u[0]<d[0]:
            rejected['backward_contact_recording_time']+=1
            continue
        matched.append((press,release,activity,d[0],u[0]))
    return matched,{'key_pair_rejections':rejections,'match_rejections':dict(rejected),
                    'complete_keys_before_matching':len(pairs)}


def rows(reader,sid,session,filename,columns):
    role='train_enrollment' if session<=8 else 'train_fit'
    with reader.open_member(sid,session,filename,expected_cohort='fit',expected_role=role,
                            purpose='feature_feasibility') as member:
        for number,row in enumerate(csv.reader(io.TextIOWrapper(member,encoding='utf-8-sig'))):
            if number>=2000000:raise ValueError('Row bound exceeded')
            if len(row)!=columns:raise ValueError('Unexpected column width')
            vals=[float(v) for v in (row[:9] if filename=='Activity.csv' else row)]
            if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite measurement')
            yield vals


def count_session(activities,keys,touches,acc,gyro):
    matched,diag=matched_pairs(keys,touches)
    eligible=0; candidate=0; exclusions=Counter(); activity_counts=[]
    for a in activities:
        if a[8] not in WRITING:continue
        aid,start,end=a[0],a[3],a[4]
        if end<start:raise ValueError('Activity end precedes start')
        pairs=[p for p in matched if p[2]==aid and a[5]<=p[0]<=p[1]<=a[6]]
        sensorrows={'acc':[r for r in acc if r[2]==aid], 'gyro':[r for r in gyro if r[2]==aid]}
        backwards={k:any(b[0]<c[0] for c,b in zip(v,v[1:])) for k,v in sensorrows.items()}
        n=int((end-start)//WIDTH)
        ac={'full_windows':n,'eligible_windows':0,'excluded':Counter()}
        for index in range(n):
            left=start+index*WIDTH;right=left+WIDTH
            candidate+=1
            pp=[p for p in pairs if left<=p[3]<right and left<=p[4]<right]
            reasons=[]
            if len(pp)<52:reasons.append('fewer_than_52_matched_keys')
            # Check chronological native contact endpoints against recorded absolute anchors.
            events=sorted([(p[0],p[3]) for p in pp]+[(p[1],p[4]) for p in pp])
            if any(b[1]<c[1] for c,b in zip(events,events[1:])):
                reasons.append('backward_matched_anchor')
            for kind,sr in sensorrows.items():
                bins=bins_complete([r[0] for r in sr],left)
                if bins['records']<3:reasons.append(kind+'_fewer_than_3_records')
                if bins['nonempty_bins']!=100:reasons.append(kind+'_empty_bins')
                if backwards[kind]:reasons.append(kind+'_activity_recording_clock_backstep')
            if reasons:
                exclusions.update(reasons);ac['excluded'].update(reasons)
            else:eligible+=1;ac['eligible_windows']+=1
        ac['excluded']=dict(ac['excluded']);activity_counts.append(ac)
    return {'candidate_full_windows':candidate,'eligible_windows':eligible,
            'excluded_windows':candidate-eligible,'exclusion_reasons_nonexclusive':dict(exclusions),
            'matched_complete_pairs':len(matched),'pair_diagnostics':diag,'activities':activity_counts}


def main():
    reader=HmogReader(phase='fit_feature_feasibility')
    selected={sid:sorted(e['session'] for e in reader.entries.values() if e['subject']==sid
             and e['session']<=16 and e['name'].endswith('/KeyPressEvent.csv') and e['uncompressed_bytes']>0)
             for sid in SUBJECTS}
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'subject_sessions':selected,
          'phase':'fit_feature_feasibility','purpose':'feature_feasibility','manifest_sha256':MANIFEST_SHA256,
          'source_sha256':digest(Path(__file__)),'guard_sha256':digest(ROOT/'scripts/hmog_data.py'),
          'pair_helper_sha256':digest(ROOT/'scripts/hmog_clock_audit.py'),
          'contract_sha256':digest(ROOT/'research/benchmarks/hmog_behaveformer_contract_proposal.md'),
          'window_ms':WIDTH,'bins':100,'complete_unique_matched_keys_min':52,'sensor_records_min':3,
          'writing_tasks':sorted(WRITING),'orientation':0,
          'bounds':'Full half-open windows anchored Activity wallstart, stop before wallend. Both touch anchors inside same window; key relative endpoints inside Activity relativebounds.',
          'pairing':'Event-time pairs, no duplicates/repeats/missing endpoints. Exact unique activity,event,action touch matches and same valid pointer contact.',
          'clock_policy':'Reject backward matched-anchor event order within window; reject windows of any activity with sensor SysTime backward step. No retiming, offset fit or relaxation.',
          'triplet_policy':'Only train_fit sessions9–16 for optimizer anchor/positive/negative; require2eligible sessions per genuineidentity and another eligible identity.',
          'limits':'2million rows/member; first4fit only; no model features, fits, selection/calibration/dev/support/test access'}
    with (OUT/'window_feasibility_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    results={}
    for sid,sessions in selected.items():
        results[sid]={}
        for session in sessions:
            data={name:list(rows(reader,sid,session,name,width)) for name,width in
                  [('Activity.csv',10),('KeyPressEvent.csv',6),('TouchEvent_im.csv',11),
                   ('Accelerometer.csv',7),('Gyroscope.csv',7)]}
            results[sid][str(session)]=count_session(data['Activity.csv'],data['KeyPressEvent.csv'],
                  data['TouchEvent_im.csv'],data['Accelerometer.csv'],data['Gyroscope.csv'])
    fit_sessions={sid:[int(s) for s,d in ss.items() if int(s)>=9 and d['eligible_windows']>0] for sid,ss in results.items()}
    positive_ids=[sid for sid,ss in fit_sessions.items() if len(ss)>=2]
    negative_ids=[sid for sid,ss in fit_sessions.items() if ss]
    triplets={sid:any(other!=sid for other in negative_ids) for sid in positive_ids}
    report={'plan':plan,'subjects':results,'eligible_train_fit_sessions':fit_sessions,
            'cross_session_triplet_possible_by_anchor_identity':triplets,
            'all_four_fit_identities_can_anchor':len(triplets)==4 and all(triplets.values()),
            'model_training_started':False,'held_out_measurements_read':False}
    with (OUT/'window_feasibility_results.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'eligible_train_fit_sessions':fit_sessions,'triplets':triplets},indent=2))

if __name__=='__main__':main()
