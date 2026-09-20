"""Bounded first-fit TRAIN schema inspection; no selection/dev/test access."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from hmog_data import HmogReader, BASE, MANIFEST_SHA256


def main():
    plan_path = BASE / 'schema_probe_plan.json'
    output = BASE / 'train_schema_probe.json'
    if plan_path.exists() or output.exists():
        raise SystemExit('Preserve existing schema inspection')
    reader = HmogReader()
    eligible = sorted(e['session'] for e in reader.entries.values()
                      if e['subject'] == '717868' and e['role'] == 'train_enrollment'
                      and e['name'].endswith('/KeyPressEvent.csv') and e['uncompressed_bytes'] > 0)
    if not eligible:
        raise SystemExit('No metadata-eligible first-fit support session')
    session = eligible[0]
    files = ['KeyPressEvent.csv', 'TouchEvent.csv', 'TouchEvent_im.csv',
             'tempTouchEvent.csv', 'Activity.csv', 'Accelerometer.csv',
             'Gyroscope.csv', 'Magnetometer.csv']
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'subject': '717868', 'session': session, 'role': 'train_enrollment',
            'selection_rule': 'First numerically ordered first-fit support session with nonzero keypress file size, metadata only',
            'files': files, 'maximum_lines_per_file': 5,
            'session_roles_sha256': MANIFEST_SHA256,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'purpose': 'TRAIN schema inspection only; no features, scores, or clock alignment claim'}
    with plan_path.open('x') as f:
        json.dump(plan, f, indent=2)
    results = {}
    for filename in files:
        member = f'717868/717868_session_{session}/{filename}'
        if member not in reader.entries:
            results[filename] = {'available': False}
            continue
        rows = list(reader.iter_lines('717868', session, filename,
                    expected_cohort='fit', expected_role='train_enrollment',
                    purpose='schema_inspection', max_lines=5))
        results[filename] = {'available': True, 'lines': rows}
    with output.open('x') as f:
        json.dump({'plan': plan, 'results': results, 'test_read': False, 'dev_read': False}, f, indent=2)
    # Only structural metadata is printed; raw rows remain private in datasets/.
    for name, result in results.items():
        lines = result.get('lines', [])
        print(name, 'available=', result['available'], 'sampled_lines=', len(lines),
              'column_counts=', [len(line.rstrip().split(',')) for line in lines])


if __name__ == '__main__':
    main()
