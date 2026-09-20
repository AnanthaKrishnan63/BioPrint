"""Strict fitting-only diagnostic of KeyRecs digraph adjacency, no model fitting."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/benchmarks/type2branch_continuity_v1'
DATA = ROOT/'research/benchmarks/data/keyrecs'
TOLERANCE = 1e-5


def parse_suffix(opaque):
    suffix = opaque.rstrip('\r\n')
    if suffix.endswith(','):
        suffix = suffix[:-1]
    prefix, *timing = suffix.rsplit(',', 5)
    keys = next(csv.reader([prefix], strict=True))
    if len(keys) != 2 or len(timing) != 5:
        raise ValueError('Invalid digraph shape')
    values = [float(x) for x in timing]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite timing')
    return keys, values


def fitting_rows(lines, allowed):
    header = next(lines)
    if not header.startswith('participant,session,key1,key2,'):
        raise ValueError('Unexpected header')
    for line in lines:
        subject, session, opaque = line.split(',', 2)
        if subject not in allowed or session != '1':
            continue
        try:
            row = parse_suffix(opaque)
        except (ValueError, csv.Error, StopIteration):
            row = None
        yield subject, row


def diagnose(rows):
    counts = Counter()
    previous = None
    run, windows = 0, 0
    maximum_errors = {'hold_plus_ud_minus_dd': 0., 'hold_plus_uu_minus_du': 0.,
                      'adjacent_hold': 0.}
    for row in rows:
        counts['rows'] += 1
        if row is None:
            counts['malformed_rows'] += 1
            windows += run//100
            run, previous = 0, None
            continue
        keys, (ht, dd, du, ud, uu) = row
        first_error, second_error = abs(ht+ud-dd), abs(ht+uu-du)
        maximum_errors['hold_plus_ud_minus_dd'] = max(maximum_errors['hold_plus_ud_minus_dd'], first_error)
        maximum_errors['hold_plus_uu_minus_du'] = max(maximum_errors['hold_plus_uu_minus_du'], second_error)
        valid = ht >= 0 and dd > 0 and first_error <= TOLERANCE and second_error <= TOLERANCE
        counts['negative_hold'] += ht < 0
        counts['nonpositive_dd'] += dd <= 0
        counts['within_row_identity_failures'] += first_error > TOLERANCE or second_error > TOLERANCE
        linked = False
        if previous is not None:
            old_keys, old_values = previous
            error = abs((old_values[2]-old_values[1])-ht)
            maximum_errors['adjacent_hold'] = max(maximum_errors['adjacent_hold'], error)
            counts['adjacency_checks'] += 1
            counts['key_mismatches'] += old_keys[1] != keys[0]
            counts['adjacent_hold_mismatches'] += error > TOLERANCE
            linked = old_keys[1] == keys[0] and error <= TOLERANCE
        if not valid or (previous is not None and not linked):
            windows += run//100
            run = 0
        if valid:
            run += 1
            previous = row
        else:
            previous = None
    windows += run//100
    return {'counts': dict(counts), 'maximum_absolute_errors': maximum_errors,
            'candidate_nonoverlapping_100row_windows': windows}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    roles_path = ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    roles = json.loads(roles_path.read_text())
    for name, expected in roles['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Role input changed')
    manifest = json.loads((DATA/'split-manifest.json').read_text())
    allowed = set(roles['roles']['fit'])
    assert len(allowed)==47 and not allowed & set(manifest['sealed_test_subjects'])
    source = DATA/'free-text.csv'
    plan = {'scope':'47 frozen fitting identities, session1 only', 'identities':sorted(allowed),
            'source_sha256':manifest['files']['free-text.csv']['sha256'],
            'role_sha256':sha(roles_path), 'script_sha256':sha(Path(__file__)),
            'absolute_tolerance_seconds':TOLERANCE,
            'tolerance_reason':'Fixed conservative floating-point equality diagnostic; not fitted',
            'not_read':['selection timings','calibration timings','DEV timings','test timings']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    if sha(source)!=plan['source_sha256']:
        raise ValueError('Raw source changed')
    grouped=defaultdict(list)
    with source.open() as stream:
        for subject,row in fitting_rows(stream, allowed):
            grouped[subject].append(row)
    reports={s:diagnose(grouped[s]) for s in sorted(allowed)}
    totals=Counter()
    for r in reports.values():totals.update(r['counts'])
    report={'status':'fitting_only_continuity_diagnostic_complete','per_identity':reports,
            'totals':dict(totals), 'eligible_N15_identities':[s for s,r in reports.items()
                if r['candidate_nonoverlapping_100row_windows']>=15],
            'limitations':['Digraph adjacency is evidence, not proof of original timestamp continuity',
                           'No raw absolute timestamps available in this CSV',
                           'No cohort selected or timing imputation performed',
                           'No model accuracy or feature-synthesis parity claim']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_identity'}))


if __name__=='__main__':main()
