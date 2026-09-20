"""TRAIN-only timing/padding diagnostics under the original schema scope."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import loadmat
from arithmetic_schema import validate_scope

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/arithmetic'
OUT = ROOT / 'research/benchmarks/arithmetic_diagnostics_v1'


def flatten(value):
    return np.asarray(value, dtype=object).reshape(-1)


def scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.size != 1 or arr.dtype.kind not in 'fiu':
        raise ValueError('Non-scalar numeric timestamp')
    x = float(arr.reshape(-1)[0])
    if not np.isfinite(x):
        raise ValueError('Nonfinite timestamp')
    return x


def inspect(data):
    image, key, correct = (flatten(data[k]) for k in ['imageT', 'keyT', 'CorrectAns'])
    if not (len(image) == len(key) == len(correct)):
        return {'error': 'unequal arrays', 'lengths': [len(image), len(key), len(correct)]}
    slots = []
    for i, (start, end, accuracy) in enumerate(zip(image, key, correct)):
        try:
            onset, press = scalar(start), scalar(end)
            slots.append(dict(index=i, onset_missing=onset is None,
                              press_missing=press is None, correct=scalar(accuracy),
                              rt=None if onset is None or press is None else press-onset))
        except ValueError as exc:
            slots.append(dict(index=i, error=str(exc), onset_shape=list(np.shape(start)),
                              press_shape=list(np.shape(end))))
    return {'slots': slots}


def main():
    schema = json.loads((DATA / 'schema_plan_v1.json').read_text())
    acquisition = json.loads((DATA / 'plan.json').read_text())
    validate_scope(schema, acquisition)
    if hashlib.sha256((DATA / 'plan.json').read_bytes()).hexdigest() != schema['acquisition_plan_sha256']:
        raise ValueError('Acquisition binding mismatch')
    OUT.mkdir(exist_ok=False)
    source = Path(__file__).read_bytes()
    (OUT / 'source.py').write_bytes(source)
    (OUT / 'plan.json').write_text(json.dumps(dict(
        schema_plan_sha256=hashlib.sha256((DATA / 'schema_plan_v1.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source).hexdigest(), scope='TRAIN-fit schema diagnostics only'), indent=2))
    records = []
    for entry in schema['entries']:
        path = DATA / 'raw' / entry['archive']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Archive binding mismatch')
        with zipfile.ZipFile(path) as archive:
            for member in entry['members']:
                if archive.getinfo(member).file_size > 2_000_000:
                    raise ValueError('Member bound exceeded')
                data = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)['Data']
                records.append(dict(subject=entry['subject'], member=member, **inspect(data)))
    slots = [slot for r in records for slot in r.get('slots', [])]
    rts = [s['rt'] for s in slots if s.get('rt') is not None]
    summary = dict(files=len(records), slot_count=len(slots),
                   padding_slots=sum(s.get('onset_missing', False) and s.get('press_missing', False) for s in slots),
                   presented_no_press=sum(not s.get('onset_missing', True) and s.get('press_missing', False) for s in slots),
                   errors=[r['member'] for r in records if 'error' in r or any('error' in s for s in r.get('slots', []))],
                   response_count=len(rts), rt_min=min(rts), rt_max=max(rts),
                   nonpositive_rt=sum(x <= 0 for x in rts), over_2p5_seconds=sum(x > 2.5 for x in rts),
                   dev_accessed=False, test_accessed=False)
    (OUT / 'report.json').write_text(json.dumps(dict(summary=summary, records=records), indent=2, allow_nan=False))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
