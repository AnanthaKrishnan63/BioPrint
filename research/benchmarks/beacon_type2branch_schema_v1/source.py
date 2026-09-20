"""TRAIN-fit-only feasibility audit for a frozen Type2Branch paired transfer."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT,sha
from beacon_neural_benchmark import mapped_key
from beacon_benchmark import read_csv

DATA=ROOT/'datasets/beacon'
OUT=ROOT/'research/benchmarks/beacon_type2branch_schema_v1'


def milliseconds(text):
    try: value=Decimal(text)*1000
    except (InvalidOperation,TypeError,ValueError) as error:raise ValueError('Invalid elapsed time') from error
    if not value.is_finite() or value!=value.to_integral_value():
        raise ValueError('Elapsed timestamp is not exact integer milliseconds')
    return int(value)


def audit_rows(rows,starts):
    parsed=[];unknown=0;disagreement=[]
    for row in rows:
        press=milliseconds(row['Elapsed Start Time']);release=milliseconds(row['Elapsed Release Time'])
        duration=milliseconds(row['Duration'])
        if release<press:raise ValueError('Negative hold')
        code=mapped_key(row['Key']);unknown+=code==0
        disagreement.append(abs(release-press-duration))
        parsed.append((press,release,code))
    if not parsed:raise ValueError('No observed keys')
    events=np.array(parsed,dtype=np.int64)
    decreases=int((np.diff(events[:,0])<0).sum())
    events=events[np.argsort(events[:,0],kind='stable')]
    # Match existing frozen BEACON float-second boundary comparisons exactly.
    times=events[:,:2]/1000.
    counts=[]
    for start in starts:
        count=int(((times[:,0]>=start)&(times[:,1]<start+30)).sum())
        if count<5:raise ValueError('Old accepted window no longer meets baseline key minimum')
        anchor=max([0]+[n for n in [25,50,75,100] if n<=count])
        counts.append({'start':start,'real_events':count,'hypothetical_anchor':anchor})
    return {'events':len(events),'unknown_codes':unknown,'unknown_fraction':unknown/len(events),
        'release_order_onset_decreases':decreases,'median_duration_disagreement_ms':float(np.median(disagreement)),
        'maximum_duration_disagreement_ms':max(disagreement),'windows':counts,
        'below25_windows':sum(r['real_events']<25 for r in counts),
        'anchor_counts':{str(n):sum(r['hypothetical_anchor']==n for r in counts) for n in [0,25,50,75,100]}}


def main():
    manifest_path=DATA/'split_manifest.json';ledger_path=DATA/'record_roles.json'
    baseline=ROOT/'research/benchmarks/beacon/frozen.json'
    previous=ROOT/'research/benchmarks/beacon-neural-v2/frozen.json'
    manifest=json.loads(manifest_path.read_text());ledger=json.loads(ledger_path.read_text())
    old=json.loads(previous.read_text());fit=json.loads(baseline.read_text())['protocol']['fit_identities']
    if fit!=['P002','P003','P006']:raise ValueError('Original fitting identities changed')
    if ledger['manifest_sha256']!=sha(manifest_path) or old['manifest_sha256']!=sha(manifest_path) or old['roles_sha256']!=sha(ledger_path):
        raise ValueError('Prior role/window provenance changed')
    roles={r['release_path']:r for r in ledger['files']}
    files=[r for r in manifest['files'] if r['participant_id'] in fit and r['modality']=='keyboard_csv']
    if len(files)!=6 or {(r['participant_id'],r['role']) for r in files}!={(s,r) for s in fit for r in ['enrollment','probe']}:
        raise ValueError('Exactly six fitting recordings required')
    for row in files:
        role=roles[row['release_path']]
        if row['split']!='train' or role['cohort']!='train' or role['record_partition']!='train' or not role['global_model_fit_allowed']:
            raise ValueError('Non-fitting observation access prohibited')
        if row['partial'] or sha(DATA/'train'/row['release_path'])!=row['sha256']:
            raise ValueError('Full keyboard recording checksum mismatch')
    sources=[Path(__file__),manifest_path,ledger_path,baseline,previous,
        ROOT/'scripts/beacon_neural_benchmark.py',ROOT/'scripts/beacon_benchmark.py',
        ROOT/'research/benchmarks/type2branch_length_pair_v1/report.json']
    OUT.mkdir(exist_ok=False)
    plan={'scope':'Only3existing BEACON fitting identities,6TRAINkeyboard recordings; no selection/calibration/DEV/test observations',
        'window_allocation':'Exact earlier dual-neural pointer-eligible window starts from frozen TRAIN audit; pointer eligibility not recomputed',
        'purpose':'Measure timing/key/length feasibility only; no embeddings, fitting, thresholds or policy selection',
        'candidate':'Frozen source-trained Type2Branch selected checkpoint, never joined across source identities',
        'prior_exposure':'Earlier BEACON DEV and KeyRecs DEV experiments disclosed; no fresh-validation claim',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'input_sha256':{str((DATA/'train'/r['release_path']).relative_to(ROOT)):r['sha256'] for r in files}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    for row in sorted(files,key=lambda r:(r['participant_id'],r['role'])):
        records=[a for a in old['train_audit'] if a['subject']==row['participant_id'] and a['role']==row['role']]
        if len(records)!=1:raise ValueError('Missing original window audit')
        starts=records[0]['eligible_window_starts']
        info=audit_rows(read_csv(DATA/'train'/row['release_path']),starts)
        results.append({'subject':row['participant_id'],'role':row['role'],**info})
    report={'status':'train_fit_schema_audit_complete','recordings':results,'metrics':{},
        'selection_calibration_dev_test_payloads_decoded':False,
        'limitations':['Stable onset sorting is an explicit source transfer adaptation',
            'Logger auto-repeat overwrites cannot be reconstructed','No model inputs or recognition scores produced',
            'Coverage anchors are descriptive, not selected enrollment or exclusion policy']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'recordings':len(results),
        'old_windows':sum(len(r['windows']) for r in results),'below25_windows':sum(r['below25_windows'] for r in results),
        'unknown_codes':sum(r['unknown_codes'] for r in results)}))


if __name__=='__main__':main()
