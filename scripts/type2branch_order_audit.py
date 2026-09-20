"""Locate incomplete fitting records and diagnose down/up ordering, no repairs."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from type2branch_continuity import ROOT, DATA, sha
from type2branch_invalid_rows import classify

OUT=ROOT/'research/benchmarks/type2branch_order_audit_v1'


def summarize(rows):
    counts=Counter()
    incomplete=[]
    for index, opaque in enumerate(rows):
        classification=classify(opaque)
        if classification['missing_timing_fields']:
            incomplete.append({'index':index,'is_first':index==0,'is_last':index==len(rows)-1,
                               'missing_fields':classification['missing_timing_fields'],
                               'empty_keys':classification['empty_key']})
        if not classification['complete_finite_timing_rows']:continue
        suffix=opaque.rstrip('\r\n')
        if suffix.endswith(','):suffix=suffix[:-1]
        _, *parts=suffix.rsplit(',',5)
        ht,dd,du,ud,uu=map(float,parts)
        assert all(math.isfinite(x) for x in [ht,dd,du,ud,uu])
        counts['complete_timing_rows']+=1
        counts['dd_negative']+=dd<0
        counts['uu_negative']+=uu<0
        counts['dd_negative_uu_nonnegative']+=dd<0 and uu>=0
        counts['dd_nonnegative_uu_negative']+=dd>=0 and uu<0
        counts['dd_and_uu_negative']+=dd<0 and uu<0
        counts['inferred_second_hold_negative']+=du-dd<0
    return {'counts':dict(counts),'incomplete_records':incomplete,'rows':len(rows)}


def main():
    OUT.mkdir(exist_ok=False)
    prior=ROOT/'research/benchmarks/type2branch_invalid_rows_v1'
    old=json.loads((prior/'plan.json').read_text())
    roles_path=ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    if sha(roles_path)!=old['role_sha256']:raise ValueError('Roles changed')
    allowed=set(json.loads(roles_path.read_text())['roles']['fit'])
    paths=[Path(__file__),ROOT/'scripts/type2branch_invalid_rows.py',ROOT/'scripts/type2branch_continuity.py']
    plan={'scope':'Same47fit IDs/session1; boundary and ordering diagnostics only',
          'role_sha256':sha(roles_path),'raw_sha256':old['source_sha256'],
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    raw=DATA/'free-text.csv'
    if sha(raw)!=old['source_sha256']:raise ValueError('Raw data changed')
    groups=defaultdict(list)
    with raw.open() as stream:
        next(stream)
        for line in stream:
            subject,session,opaque=line.split(',',2)
            if subject in allowed and session=='1':groups[subject].append(opaque)
    per={s:summarize(groups[s]) for s in sorted(allowed)}
    total=Counter()
    for r in per.values():total.update(r['counts'])
    incomplete=[v for r in per.values() for v in r['incomplete_records']]
    baseline=json.loads((prior/'report.json').read_text())['totals']
    assert sum(r['rows'] for r in per.values())==baseline['rows']
    assert total['complete_timing_rows']==baseline['complete_finite_timing_rows']
    assert total['dd_negative']==baseline['dd_negative_all_finite']
    report={'status':'fitting_only_order_diagnostic_complete','totals':dict(total),
            'incomplete_records':len(incomplete),'incomplete_at_end':sum(x['is_last'] for x in incomplete),
            'incomplete_at_start':sum(x['is_first'] for x in incomplete),'per_identity':per,
            'limitations':['Interval signs do not prove recorder ordering policy',
                           'No rows repaired, removed, sorted or model-ready sequences emitted',
                           'Selection/calibration/DEV/test observation values not decoded']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_identity'}))


if __name__=='__main__':main()
