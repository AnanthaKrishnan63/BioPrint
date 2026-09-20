"""Classify previously flagged fitting-only records without repairing data."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from type2branch_continuity import ROOT, DATA, parse_suffix, sha

OUT = ROOT/'research/benchmarks/type2branch_invalid_rows_v1'


def classify(opaque):
    suffix = opaque.rstrip('\r\n')
    if suffix.endswith(','):suffix = suffix[:-1]
    parts = suffix.rsplit(',',5)
    result = Counter()
    if len(parts)!=6:
        return Counter({'wrong_field_count':1})
    try:
        keys=next(csv.reader([parts[0]],strict=True))
        if len(keys)!=2:result['wrong_key_count']+=1
        if any(not key for key in keys):result['empty_key']+=1
    except (csv.Error,StopIteration):
        result['key_csv_error']+=1
    values=[]
    for value in parts[1:]:
        if not value:
            result['missing_timing_fields']+=1
            continue
        try: number=float(value)
        except ValueError:
            result['nonnumeric_timing_fields']+=1
            continue
        if not math.isfinite(number):result['nonfinite_timing_fields']+=1
        else:values.append(number)
    if len(values)==5:
        result['complete_finite_timing_rows']+=1
        result['dd_zero_all_finite']+=values[1]==0
        result['dd_negative_all_finite']+=values[1]<0
        # No rounding performed: this only measures exact wire compatibility.
        result['submillisecond_timing_fields']+=sum(abs(v*1000-round(v*1000))>1e-6 for v in values)
    try:
        _,timing=parse_suffix(opaque)
        result['prior_decodable_rows']+=1
        result['dd_zero_prior_decodable']+=timing[1]==0
        result['dd_negative_prior_decodable']+=timing[1]<0
    except (ValueError,csv.Error,StopIteration):
        result['prior_undecodable_rows']+=1
    return result


def main():
    OUT.mkdir(exist_ok=False)
    prior_dir=ROOT/'research/benchmarks/type2branch_continuity_v1'
    prior_plan=json.loads((prior_dir/'plan.json').read_text())
    prior=json.loads((prior_dir/'report.json').read_text())
    roles_path=ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    if sha(roles_path)!=prior_plan['role_sha256']:
        raise ValueError('Role assignment changed')
    allowed=set(json.loads(roles_path.read_text())['roles']['fit'])
    if allowed!=set(prior_plan['identities']):raise ValueError('Fit identities changed')
    plan={'scope':'Classification only; same47fitting identities/session1; no repairs',
          'source_sha256':prior_plan['source_sha256'], 'role_sha256':sha(roles_path),
          'prior_report_sha256':sha(prior_dir/'report.json'),
          'script_sha256':sha(Path(__file__)),
          'parser_sha256':sha(ROOT/'scripts/type2branch_continuity.py'),
          'integer_ms_tolerance':1e-6}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    source=DATA/'free-text.csv'
    if sha(source)!=plan['source_sha256']:raise ValueError('Raw source changed')
    per={s:Counter() for s in sorted(allowed)}
    with source.open() as stream:
        next(stream)
        for line in stream:
            subject,session,opaque=line.split(',',2)
            if subject not in allowed or session!='1':continue
            per[subject].update(classify(opaque));per[subject]['rows']+=1
    total=Counter()
    for counts in per.values():total.update(counts)
    assert total['rows']==prior['totals']['rows']
    assert total['prior_undecodable_rows']==prior['totals']['malformed_rows']
    assert total['dd_zero_prior_decodable']+total['dd_negative_prior_decodable']==prior['totals']['nonpositive_dd']
    report={'status':'fitting_only_invalid_record_classification_complete','totals':dict(total),
            'per_identity':{s:dict(c) for s,c in per.items()},
            'limitations':['Timing-unit assumption inherited from prior KeyRecs protocol',
                           'Fractional milliseconds cannot enter strict integer CSV without an explicit adaptation',
                           'No repaired keys, imputation, clipping or changed eligibility',
                           'Selection/calibration/DEV/test observations not decoded']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'totals':dict(total)}))


if __name__=='__main__':main()
