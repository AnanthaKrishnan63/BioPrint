"""Predeclared frozen TypeNet + SapiMouse transfer on real paired BEACON windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import beacon_neural_benchmark as base
from engine.pointer_sequence import PointerSequenceEncoder

OUT = base.ROOT / 'research/benchmarks/beacon-neural-v2'
POINTER = base.ROOT / 'research/benchmarks/pointer_sapimouse/encoder.torchscript.pt'
POINTER_HASH = 'a0e613d7029a42f4e25177c75e33b2ecd551a8e1571720df0e9ee0bafb21a32a'
FEATURES = {'typenet': [0], 'mouse': list(range(1, 17)),
            'handcrafted_behavior': list(range(1, 33)), 'sapimouse': [33],
            'dual_neural': [0, 33], 'hybrid_behavior_fusion': list(range(34))}
PROTOCOL = dict(base.PROTOCOL, pointer_model_sha256=POINTER_HASH,
    representation='frozen source-trained TypeNet and SapiMouse FCN; neither encoder updates on BEACON',
    window_rule='exact baseline 30-second windows restricted to at least129 raw mouse records; same eligible windows for every comparator',
    pointer_transform='all event types in CSV order; abs coordinate differences; nonoverlap128 differences; joint256-value population z-score; transpose2x128',
    pointer_window_embedding='mean of all complete128-difference block embeddings inside each30second window; no cross-window joins or padding',
    pointer_gallery='first up to5 eligible training enrollment windows; mean Euclidean distance',
    models=FEATURES,
    decision_history='predeclared source-transfer extension; no choices based on v1 dev scores or SapiMouse dev outcomes',
    availability_rule='at least2 eligible enrollment windows and at least1 eligible probe window per identity; report all exclusions')
ORIGINAL_LOAD, ORIGINAL_PAIRS = base.load, base.pairs


def load(cohort, encoder):
    if cohort not in ('train', 'dev'):
        raise ValueError('Test is sealed')
    if base.checksum(POINTER) != POINTER_HASH:
        raise ValueError('Frozen pointer encoder hash mismatch')
    data, audits = ORIGINAL_LOAD(cohort, encoder)
    manifest, _ = base.manifests()
    pointer = PointerSequenceEncoder(POINTER)
    for subject, records in data.items():
        for role, record in records.items():
            row = next(r for r in manifest['files'] if r['split'] == cohort and
                       r['participant_id'] == subject and r['role'] == role and r['modality'] == 'mouse_csv')
            rows = base.baseline.read_csv(base.DATA / cohort / row['release_path'], row['partial'])
            raw = np.array([[float(r['Elapsed Time']), float(r['X']), float(r['Y'])] for r in rows])
            if not np.isfinite(raw).all() or (np.diff(raw[:, 0]) < 0).any():
                raise ValueError('Pointer elapsed clock invalid; no inferred reordering')
            eligible, blocks, starts = [], [], []
            for window in record['windows']:
                start = window['start']
                points = raw[(raw[:, 0] >= start) & (raw[:, 0] < start + 30), 1:]
                embeddings = pointer.encode(points)
                if not len(embeddings):
                    continue
                window['pointer_embedding'] = embeddings.mean(axis=0).tolist()
                eligible.append(window); blocks.append(len(embeddings)); starts.append(float(start))
            for audit in audits:
                if audit['subject'] == subject and audit['role'] == role:
                    audit.update(pointer_eligible_windows=len(eligible), pointer_dropped_windows=len(record['windows']) - len(eligible),
                                 pointer_blocks_per_window=blocks, eligible_window_starts=starts)
            record['windows'] = eligible
    valid = {s: d for s, d in data.items() if len(d['enrollment']['windows']) >= 2 and d['probe']['windows']}
    for audit in audits:
        audit['identity_eligible'] = audit['subject'] in valid
    return valid, audits


def pairs(data):
    x, y, groups = ORIGINAL_PAIRS(data)
    galleries = {owner: np.array([w['pointer_embedding'] for w in records['enrollment']['windows'][:5]])
                 for owner, records in data.items()}
    pointer_distances = []
    for owner, actual, start in groups:
        window = next(w for w in data[actual]['probe']['windows'] if str(w['start']) == start)
        pointer_distances.append(np.linalg.norm(galleries[owner] - np.array(window['pointer_embedding']), axis=1).mean())
    return np.column_stack([x, pointer_distances]), y, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preregister', 'train', 'dev'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = OUT / 'declaration.json'
    if args.stage == 'preregister':
        with declaration.open('x') as f:
            json.dump(PROTOCOL, f, indent=2)
        print('Predeclared frozen dual-encoder transfer before evaluation')
        return
    if json.loads(declaration.read_text()) != PROTOCOL:
        raise ValueError('Preregistered protocol changed')
    base.load, base.pairs, base.FEATURES, base.PROTOCOL = load, pairs, FEATURES, PROTOCOL
    if args.stage == 'train':
        base.train(OUT)
        (OUT / 'source.py').write_text(Path(__file__).read_text())
        (OUT / 'shared_source.py').write_text(Path(base.__file__).read_text())
    else:
        if (OUT / 'dev_results.json').exists():
            raise SystemExit('Preserve existing dev results')
        base.validate(OUT)


if __name__ == '__main__':
    main()
