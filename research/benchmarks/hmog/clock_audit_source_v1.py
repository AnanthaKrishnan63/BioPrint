"""Frozen first-fit TRAIN clock/contact audit; no extraction, models or scores."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import math
import numpy as np
from hmog_data import HmogReader, MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/hmog'
SUBJECT = '717868'
FILES = ['KeyPressEvent.csv', 'TouchEvent.csv', 'TouchEvent_im.csv',
         'tempTouchEvent.csv', 'Accelerometer.csv', 'Gyroscope.csv', 'Magnetometer.csv']


def complete_key_pairs(rows):
    """Order by event time; drop duplicate events and ambiguous/missing cycles.

    Rows have sys,event,activity,action,key,orientation. A duplicate marks its
    entire press cycle invalid, even if a later release exists.
    """
    counts = Counter(tuple(r[1:]) for r in rows)
    ordered = sorted({tuple(r[1:]) for r in rows}, key=lambda r: (r[0], r[2]))
    active, pairs = {}, []
    reasons = Counter()
    for event, activity, action, key, orientation in ordered:
        identity = (activity, key, orientation)
        duplicate = counts[(event, activity, action, key, orientation)] > 1
        if duplicate:
            reasons['duplicate_events'] += counts[(event, activity, action, key, orientation)] - 1
        if action == 0:
            if identity in active:
                reasons['repeated_down'] += 1
                active[identity] = (event, False)
            else:
                active[identity] = (event, not duplicate)
        elif action == 1:
            down = active.pop(identity, None)
            if down is None:
                reasons['missing_down'] += 1
            elif down[1] and not duplicate and event > down[0]:
                pairs.append((down[0], event, activity, key, orientation))
            else:
                reasons['invalid_cycle'] += 1
        else:
            reasons['unknown_action'] += 1
    reasons['missing_up'] = len(active)
    return pairs, dict(reasons)


def summary(values):
    if not values:
        return {'count': 0}
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'min': float(a.min()), 'median': float(np.median(a)),
            'p01': float(np.quantile(a, .01)), 'p99': float(np.quantile(a, .99)),
            'max': float(a.max()), 'std': float(a.std())}


def read_rows(reader, session, filename, columns):
    role = 'train_enrollment' if session <= 8 else 'train_fit'
    rows, invalid = [], Counter()
    with reader.open_member(SUBJECT, session, filename, expected_cohort='fit',
                            expected_role=role, purpose='schema_inspection') as member:
        for number, row in enumerate(csv.reader(io.TextIOWrapper(member, encoding='utf-8-sig'))):
            if number >= 2000000:
                raise ValueError('Audit row bound exceeded')
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


def stream_stats(rows, invalid, kind):
    result = {'valid_numeric_rows': len(rows), 'invalid_rows': invalid,
              'logging_timestamp_backsteps': sum(b[0] < a[0] for a, b in zip(rows, rows[1:])),
              'event_timestamp_backsteps_in_file_order': sum(b[1] < a[1] for a, b in zip(rows, rows[1:]))}
    result['sys_minus_event_ms'] = summary([r[0] - r[1] / (1e6 if kind == 'sensor' else 1) for r in rows])
    if kind != 'sensor':
        action_col = 3 if kind == 'key' else 5
        allowed = {0, 1} if kind == 'key' else {0, 1, 2, 3, 4, 5, 6}
        result['action_counts'] = {str(k): v for k, v in Counter(r[action_col] for r in rows).items()}
        result['unknown_action_rows'] = sum(r[action_col] not in allowed for r in rows)
        result['noninteger_activity_or_action'] = sum(r[2] != int(r[2]) or r[action_col] != int(r[action_col]) for r in rows)
        if kind == 'touch':
            result['noninteger_count_or_pointer'] = sum(r[3] != int(r[3]) or r[4] != int(r[4]) for r in rows)
            result['negative_contact_count'] = sum(r[3] < 0 for r in rows)
            result['negative_pressure_or_size'] = sum(r[8] < 0 or r[9] < 0 for r in rows)
    return result


def anchors(rows, kind):
    result = Counter()
    for r in rows:
        action = r[3] if kind == 'key' else r[5]
        if action in (0, 5):
            result[(r[2], r[1], 0)] += 1
        elif action in (1, 6):
            result[(r[2], r[1], 1)] += 1
    return result


def main():
    reader = HmogReader()
    sessions = sorted(e['session'] for e in reader.entries.values()
                      if e['subject'] == SUBJECT and e['session'] <= 16
                      and e['name'].endswith('/KeyPressEvent.csv') and e['uncompressed_bytes'] > 0)
    OUT.mkdir(exist_ok=True, parents=True)
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'subject': SUBJECT,
            'sessions': sessions, 'purpose': 'schema_inspection', 'files': FILES,
            'selection': 'All metadata-nonempty keypress sessions1–16 of first fit identity only',
            'row_bound_per_member': 2000000, 'manifest_sha256': MANIFEST_SHA256,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'key_columns': ['SysTime', 'PressTime', 'ActivityID', 'PressType', 'KeyID', 'orientation'],
            'touch_columns': ['SysTime', 'eventTime', 'ActivityID', 'count', 'pointerID', 'action', 'x', 'y', 'pressure', 'size', 'orientation'],
            'clock_scope': 'Offsets and spreads only; sys-event and sys-sensor/1e6 are not proof of equal clock origin. No uptime/elapsedRealtime equivalence assumed.',
            'pair_rule': 'Stable event-time order; reject duplicate/ambiguous/missing/nonpositive cycles. Never synthesize events.',
            'anchor_rule': 'Exact(activityID,eventTime,action), touch0/5 normalized down and1/6 up; no tolerance fit',
            'no_model_fit': True, 'no_dev_calibration_selection_test_read': True}
    with (OUT / 'clock_audit_plan.json').open('x') as f:
        json.dump(plan, f, indent=2)
    results = {}
    for session in sessions:
        key, bad = read_rows(reader, session, 'KeyPressEvent.csv', 6)
        pairs, dropped = complete_key_pairs(key)
        key_anchors = anchors(key, 'key')
        item = {'key': stream_stats(key, bad, 'key'), 'complete_key_pairs': len(pairs),
                'pair_rejections': dropped, 'key_hold_ms': summary([p[1]-p[0] for p in pairs]), 'streams': {}}
        touch_anchors, touch_offsets = {}, {}
        for filename in FILES[1:]:
            name = f'{SUBJECT}/{SUBJECT}_session_{session}/{filename}'
            if name not in reader.entries:
                item['streams'][filename] = {'available': False}
                continue
            kind = 'sensor' if filename in FILES[4:] else 'touch'
            rows, invalid = read_rows(reader, session, filename, 6 if kind == 'sensor' else 11)
            stat = stream_stats(rows, invalid, kind)
            if kind == 'touch':
                aa = anchors(rows, kind); touch_anchors[filename] = aa
                contact_rows = [[r[0], r[1], r[2], 0 if r[5] in (0, 5) else 1, r[4], r[10]]
                                for r in rows if r[5] in (0, 1, 5, 6)]
                contacts, contact_rejections = complete_key_pairs(contact_rows)
                stat['complete_touch_contacts'] = len(contacts)
                stat['contact_rejections'] = contact_rejections
                touch_offsets[filename] = stat['sys_minus_event_ms'].get('median')
                stat['duplicate_contact_anchors'] = sum(v-1 for v in aa.values())
                stat['key_anchor_unique_total'] = len(key_anchors)
                stat['key_anchor_exact_matches'] = len(key_anchors.keys() & aa.keys())
                stat['key_anchor_coverage'] = len(key_anchors.keys() & aa.keys()) / len(key_anchors) if key_anchors else None
                stat['key_activity_overlap_count'] = len({r[2] for r in key} & {r[2] for r in rows})
            item['streams'][filename] = stat
        item['touch_pairwise_exact_anchor_overlap'] = {
            a+' vs '+b: len(touch_anchors[a].keys() & touch_anchors[b].keys())
            for a in touch_anchors for b in touch_anchors if a < b}
        item['sensor_minus_touch_median_offset_ms'] = {
            sensor+' vs '+touch: stat['sys_minus_event_ms']['median']-offset
            for sensor, stat in item['streams'].items() if sensor in FILES[4:] and stat.get('valid_numeric_rows', 0)
            for touch, offset in touch_offsets.items() if offset is not None}
        results[str(session)] = item
    with (OUT / 'clock_audit_results.json').open('x') as f:
        json.dump({'plan': plan, 'sessions': results, 'limitations': [
            'One fit subject only. No authentication evidence.',
            'Exact anchors establish correspondence only where observed; absent matches do not establish desynchronization.',
            'Offset medians cannot identify clock origin or authorize dev retiming.',
            'No raw rows or key codes persisted in this aggregate report.']}, f, indent=2)
    print(json.dumps({'sessions': sessions, 'complete_key_pairs': {s:r['complete_key_pairs'] for s,r in results.items()}}, indent=2))


if __name__ == '__main__':
    main()
