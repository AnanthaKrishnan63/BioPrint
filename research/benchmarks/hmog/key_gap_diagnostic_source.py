"""First-fit session3 TRAIN key-event gap counts, no features or model fit."""
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from hmog_data import HmogReader,MANIFEST_SHA256
from hmog_clock_audit import complete_key_pairs
from hmog_fit_feature_extract_v2 import strict_pairs,WRITING

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/hmog'


def main():
    reader=HmogReader()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'subject':'717868','session':3,
          'phase':'first_fit_schema','purpose':'schema_inspection','manifest_sha256':MANIFEST_SHA256,
          'source_sha256':digest(Path(__file__)),
          'dependencies':{n:digest(ROOT/'scripts'/n) for n in ['hmog_data.py','hmog_clock_audit.py','hmog_fit_feature_extract_v2.py']},
          'scope':'Counts only in original Activity anchored30s windows; classify extra native key events in first52 strict matchedpair span. No extraction model orheldoutread.'}
    with (OUT/'key_gap_diagnostic_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    with (OUT/'key_gap_diagnostic_source.py').open('x') as f:f.write(Path(__file__).read_text())
    data={}
    for name in ['Activity.csv','KeyPressEvent.csv','TouchEvent_im.csv']:
        with reader.open_member('717868',3,name,expected_cohort='fit',expected_role='train_enrollment',purpose='schema_inspection') as m:
            rows=[]
            for n,r in enumerate(csv.reader(io.TextIOWrapper(m,encoding='utf-8-sig'))):
                if n>=2000000:raise ValueError('Bound')
                rows.append([float(v) for v in (r[:9] if name=='Activity.csv' else r)])
            data[name]=rows
    keys=data['KeyPressEvent.csv'];matched=strict_pairs(keys,data['TouchEvent_im.csv'])
    allpairs,pairreject=complete_key_pairs(keys)
    endpoint_pairs={}
    for p,u,a,k,o in allpairs:
        endpoint_pairs[(p,a,0,k,o)]=(p,u,a,k,o)
        endpoint_pairs[(u,a,1,k,o)]=(p,u,a,k,o)
    output=[];aggregate=Counter()
    for activity in data['Activity.csv']:
        if activity[8] not in WRITING:continue
        for index in range(int((activity[4]-activity[3])//30000)):
            left=activity[3]+index*30000;right=left+30000;aid=activity[0]
            pp=[p for p in matched if p[2]==aid and left<=p[4]<right and left<=p[5]<right][:52]
            record={'window_index':index,'retained_pairs':len(pp)}
            if not pp:
                output.append(record);continue
            lo=min(p[0] for p in pp);hi=max(p[1] for p in pp)
            raw=Counter(tuple(r[1:]) for r in keys if r[2]==aid and lo<=r[1]<=hi)
            expected=Counter()
            for p,u,a,k,_,_ in pp:
                expected[(p,a,0,k,0)]+=1;expected[(u,a,1,k,0)]+=1
            extra=raw-expected;counts=Counter()
            for event,count in extra.items():
                if event in expected:label='duplicate_retained_endpoint'
                elif event in endpoint_pairs:
                    pair=endpoint_pairs[event]
                    label='overlap_boundary_extra_endpoint' if pair[0]<lo or pair[1]>hi else 'unmatched_complete_cycle_endpoint'
                elif event[2] not in (0,1):label='unknown_action_extra_event'
                else:label='orphan_or_ambiguous_extra_event'
                counts[label]+=count
            record.update({'raw_events_in_retained_span':sum(raw.values()),
                'retained_endpoint_ratio':sum(expected.values())/sum(raw.values()),
                'extra_event_categories':dict(counts),'exact_raw_counter_match':raw==expected,
                'has_required52':len(pp)==52})
            aggregate.update(counts);output.append(record)
    ratios=[w['retained_endpoint_ratio'] for w in output if 'retained_endpoint_ratio' in w]
    report={'plan':plan,'windows':output,'extra_event_category_totals':dict(aggregate),
            'complete_key_pair_rejections':pairreject,'ratio_summary':{k:float(np.quantile(ratios,q)) for k,q in [('min',0),('median',.5),('max',1)]},
            'limitations':['Endpoint categories are structural, not evidence that absent events were intended keyboard presses.',
                           'Span usesfirst52 strictmatchedpairs; shorterwindowsreportedseparately. Raw key IDs/rows not exported.'],
            'held_out_measurements_read':False,'feature_arrays_or_models_created':False}
    with (OUT/'key_gap_diagnostic_results.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'categories':dict(aggregate),'ratios':report['ratio_summary'],'windows':len(output)}))

if __name__=='__main__':main()
