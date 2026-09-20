"""All-fit TRAIN structural consistency audit; no modeling or row exports."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from hmog_data import HmogReader, MANIFEST_SHA256
from hmog_clock_audit import complete_key_pairs, anchors, stream_stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/hmog'
SUBJECTS = ['717868', '526319', '986737', '539502']
WRITING = {3, 4, 9, 10, 15, 16, 21, 22}


def rows_for(reader, sid, session, filename):
    role = 'train_enrollment' if session <= 8 else 'train_fit'
    with reader.open_member(sid, session, filename, expected_cohort='fit',
                            expected_role=role, purpose='schema_inspection') as member:
        for count, row in enumerate(csv.reader(io.TextIOWrapper(member, encoding='utf-8-sig'))):
            if count >= 2000000:
                raise ValueError('Row bound exceeded')
            yield row


def numeric(reader, sid, session, filename, columns):
    rows, invalid = [], Counter()
    for row in rows_for(reader, sid, session, filename):
        if len(row) != columns:
            invalid['column_count'] += 1
            continue
        try:
            values = [float(v) for v in row]
        except ValueError:
            invalid['nonnumeric'] += 1
            continue
        if not all(math.isfinite(v) for v in values):
            invalid['nonfinite'] += 1
            continue
        rows.append(values)
    return rows, dict(invalid)


def coverage(rows, activities, orientation_col):
    missing, outside, portrait, writing, eligible = 0, 0, 0, 0, 0
    for row in rows:
        a = activities.get(row[2])
        portrait += row[orientation_col] == 0
        if a is None:
            missing += 1
            continue
        outside += not a[5] <= row[1] <= a[6]
        writing += a[8] in WRITING
        eligible += a[8] in WRITING and row[orientation_col] == 0 and a[5] <= row[1] <= a[6]
    return {'total_rows': len(rows), 'unknown_activity_rows': missing,
            'outside_activity_relative_bounds': outside,
            'portrait_rows': portrait, 'writing_task_rows': writing,
            'portrait_writing_in_bounds_rows': eligible,
            'orientation_counts': dict(Counter(str(r[orientation_col]) for r in rows)),
            'unknown_orientation_rows': sum(r[orientation_col] not in (0, 1) for r in rows),
            'activity_membership_pass': missing == 0, 'relative_bounds_pass': outside == 0}


def main():
    reader = HmogReader(phase='all_fit_schema')
    selected = {sid: sorted(e['session'] for e in reader.entries.values()
                if e['subject'] == sid and e['session'] <= 16 and
                e['name'].endswith('/KeyPressEvent.csv') and e['uncompressed_bytes'] > 0)
                for sid in SUBJECTS}
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'phase': 'all_fit_schema', 'subject_sessions': selected,
            'role_manifest_sha256': MANIFEST_SHA256,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'guard_sha256': hashlib.sha256((ROOT/'scripts/hmog_data.py').read_bytes()).hexdigest(),
            'pair_helpers_sha256': hashlib.sha256((ROOT/'scripts/hmog_clock_audit.py').read_bytes()).hexdigest(),
            'activity_columns': ['ID','subject','session','wall_start','wall_end','relative_start','relative_end','scenario','task','content'],
            'writing_task_ids': sorted(WRITING), 'portrait_orientation': 0,
            'known_orientation_codes': [0,1], 'row_limit_per_member': 2000000,
            'checks': ['activity membership and relative bounds', 'full numeric-row multiset temp equals ordinary plus im',
                       'key-event pairing and exact contact anchors', 'logging timestamp offsets, no retiming'],
            'no_models_features_dev_or_test': True}
    with (OUT/'fit_consistency_plan.json').open('x') as f:
        json.dump(plan, f, indent=2)
    results = {}
    for sid, sessions in selected.items():
        results[sid] = {}
        for session in sessions:
            activities, activity_invalid = {}, Counter()
            for row in rows_for(reader, sid, session, 'Activity.csv'):
                if len(row) != 10:
                    activity_invalid['column_count'] += 1
                    continue
                try:
                    a = [float(v) for v in row[:9]]
                except ValueError:
                    activity_invalid['nonnumeric_metadata'] += 1
                    continue
                if not all(math.isfinite(v) for v in a):
                    activity_invalid['nonfinite'] += 1
                    continue
                if a[0] in activities:
                    activity_invalid['duplicate_activity_id'] += 1
                if a[1] != int(sid) or a[2] != session:
                    activity_invalid['subject_session_mismatch'] += 1
                activities[a[0]] = a
            key, bad = numeric(reader,sid,session,'KeyPressEvent.csv',6)
            pairs, dropped = complete_key_pairs(key)
            ka = anchors(key,'key')
            item = {'activity_count':len(activities), 'activity_invalid':dict(activity_invalid),
                    'key':{**stream_stats(key,bad,'key'), **coverage(key,activities,5)},
                    'complete_key_pairs':len(pairs), 'pair_rejections':dropped,'streams':{}}
            touches = {}
            for filename in ['TouchEvent.csv','TouchEvent_im.csv','tempTouchEvent.csv',
                             'Accelerometer.csv','Gyroscope.csv','Magnetometer.csv']:
                name = f'{sid}/{sid}_session_{session}/{filename}'
                if name not in reader.entries:
                    item['streams'][filename] = {'available':False}
                    continue
                kind = 'touch' if 'TouchEvent' in filename else 'sensor'
                rows,bad = numeric(reader,sid,session,filename,11 if kind=='touch' else 7)
                stat = stream_stats(rows,bad,kind)
                if kind == 'touch':
                    touches[filename] = Counter(tuple(r) for r in rows)
                    aa = anchors(rows,'touch')
                    stat.update(coverage(rows,activities,10))
                    stat['key_anchor_coverage'] = len(ka.keys() & aa.keys())/len(ka) if ka else None
                    stat['duplicate_anchors'] = sum(v-1 for v in aa.values())
                    contact = [[r[0],r[1],r[2],0 if r[5] in (0,5) else 1,r[4],r[10]]
                               for r in rows if r[5] in (0,1,5,6)]
                    cp, cd = complete_key_pairs(contact)
                    stat['complete_contacts'],stat['contact_rejections'] = len(cp),cd
                item['streams'][filename] = stat
            if len(touches)==3:
                combined=touches['TouchEvent.csv']+touches['TouchEvent_im.csv']
                temp=touches['tempTouchEvent.csv']
                item['temp_full_numeric_row_multiset_check'] = {'equal':temp==combined,
                     'temp_excess_rows':sum((temp-combined).values()),
                     'temp_missing_rows':sum((combined-temp).values()),
                     'qualification':'Numeric equality of all11 columns, including multiplicities; not byte-string formatting identity.'}
            results[sid][str(session)] = item
    document={'plan':plan,'subjects':results,'interpretation':[
        'All-fit TRAIN schema evidence only, no model performance.',
        'Shared recorded wall-clock timestamps permit an alternative observable timeline, but logging delay and batching can distort event-local alignment.',
        'Keep key hold times on relative event time. Do not equate uptime with sensor elapsedRealtime or derive a dev offset.',
        'Exact key-touch event anchors can locate corresponding touch SysTime; this does not synchronize unmatched sensor samples or prove a common physical event instant.',
        'No raw rows, key identifiers, content, or measurement arrays exported.']}
    with (OUT/'fit_consistency_results.json').open('x') as f:
        json.dump(document,f,indent=2)
    print(json.dumps({'subjects':list(results),'sessions':sum(map(len,results.values()))}))

if __name__=='__main__':
    main()
