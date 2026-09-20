"""Frozen TypeNet representation transfer and genuinely paired BEACON fusion.

The KeyRecs-trained network is never updated here. Fusion fits/selects/calibrates
on BEACON training identities; personal enrollment is explicitly training data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data_keystrokes import checksum
from typenet_benchmark import TypeNet, TimingTransform, keycode
import beacon_benchmark as baseline
from eval.strict_cmu import threshold_for, eer

DATA = ROOT / 'datasets/beacon'
SOURCE = ROOT / 'research/benchmarks/results/typenet-keyrecs-v1'
OUT = ROOT / 'research/benchmarks/beacon-neural-v1'
SPECIAL = {'alt': 18, 'alt_l': 18, 'alt_r': 18, 'alt_gr': 225, 'ctrl': 17,
           'ctrl_l': 17, 'ctrl_r': 17, 'shift': 16, 'shift_l': 16, 'shift_r': 16,
           'space': 32, 'tab': 9, 'enter': 13, 'backspace': 8, 'caps_lock': 20,
           'esc': 27, 'cmd': 91, 'cmd_l': 91, 'cmd_r': 92, 'left': 37, 'up': 38,
           'right': 39, 'down': 40, 'home': 36, 'end': 35, 'page_up': 33, 'page_down': 34,
           'delete': 46, 'insert': 45, 'num_lock': 144, 'scroll_lock': 145,
           'media_volume_up': 175, 'media_volume_down': 174, 'media_volume_mute': 173}
FEATURES = {'typenet': [0], 'mouse': list(range(1, 17)),
            'neural_behavior_fusion': list(range(17)),
            'hybrid_behavior_fusion': list(range(33))}
PROTOCOL = {
    'representation': 'frozen KeyRecs TypeNet; no encoder fitting on BEACON',
    'source_model_sha256': checksum(SOURCE / 'model.pt'),
    'paired_window_seconds': 30, 'pairing': 'same participant, recording and elapsed-time window',
    'window_rule': 'reuse every exact baseline accepted30s window; no neural quality-based reselection',
    'sequence_rule': 'first at most50 adjacent key pairs entirely inside window; keep at least4 pairs',
    'short_sequences': 'gather last valid recurrent state; frozen BN means later padding cannot influence it',
    'keycode_mapping': 'letters/digits/ASCII plus explicit pynput special-key table; unknown0; reject unknown>1%',
    'clock': 'Elapsed Start/Release Time seconds, confirmed against source logger and Duration',
    'clock_gate': 'reject any negative hold or median(abs(release-start-Duration))>.01seconds',
    'encoder_gallery': 'first up to5 enrollment windows per account, mean Euclidean distance',
    'mouse': 'unchanged16 baseline summary features, per-account training-enrollment median/MAD distance',
    'fusion_candidates_C': [.01, .1, 1.],
    'selection': 'same fit/selection/calibration identity groups as frozen baseline; dev untouched until freeze',
    'support_roles': 'record_roles.json personal_enrollment is training, including support for dev identities',
    'test_accessed': False,
    'limitations': ['gameplay rather than typing/login',
       'source logger can overwrite press time on key auto-repeat; holds are recorder-defined',
       'key labels are mapped, not original browser event.keyCode; modifier side distinctions collapse',
       'short sequences differ from full50-event source training; no SOTA reproduction claim',
       'few independent people; comparison counts do not create independent trials'],
}


def mapped_key(label):
    if label.startswith('key.'):
        name = label[4:]
        if name in SPECIAL:
            return SPECIAL[name]
        match = re.fullmatch(r'f([1-9]|1[0-9]|2[0-4])', name)
        return 111 + int(match[1]) if match else 0
    match = re.fullmatch(r'<([0-9]+)>', label)
    if match:
        code = int(match[1])
        return code if 0 <= code <= 255 else 0
    if len(label) == 1 and 1 <= ord(label) <= 26:
        return ord(label) + 64  # Ctrl+A ... Ctrl+Z control-character convention.
    return keycode(label)


class Encoder:
    def __init__(self):
        torch.set_num_threads(2)
        config = json.loads((SOURCE / 'frozen.json').read_text())
        if checksum(SOURCE / 'model.pt') != config['model_sha256']:
            raise ValueError('Source model hash mismatch')
        self.model = TypeNet().eval()
        self.model.load_state_dict(torch.load(SOURCE / 'model.pt', weights_only=True))
        self.transform = TimingTransform()
        with np.load(SOURCE / 'frozen-arrays.npz', allow_pickle=False) as a:
            self.transform.median, self.transform.lo, self.transform.hi = a['median'], a['lo'], a['hi']

    def embed(self, sequences):
        lengths = np.array([len(s) for s in sequences])
        if (lengths < 1).any() or (lengths > 50).any():
            raise ValueError('Sequence length outside encoder contract')
        x = np.zeros((len(sequences), 50, 5), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            x[i, :len(sequence)] = sequence
        x = self.transform.apply(x)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                a = torch.as_tensor(x[start:start + 128])
                m = self.model
                a = m.input_bn(a.transpose(1, 2)).transpose(1, 2)
                a = m.lstm1(a)
                a = m.middle_bn(a.transpose(1, 2)).transpose(1, 2)
                a = m.lstm2(m.dropout(a))
                index = torch.as_tensor(lengths[start:start + len(a)] - 1)
                outputs.append(a[torch.arange(len(a)), index].numpy())
        return np.concatenate(outputs)


def manifests():
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    ledger = json.loads((DATA / 'record_roles.json').read_text())
    if ledger['manifest_sha256'] != checksum(DATA / 'split_manifest.json'):
        raise ValueError('Support ledger is not bound to current dataset manifest')
    return manifest, {row['release_path']: row for row in ledger['files']}


def load(cohort, encoder):
    if cohort not in ('train', 'dev'):
        raise ValueError('Test is sealed')
    manifest, ledger = manifests()
    grouped = defaultdict(list)
    for row in manifest['files']:
        if row['split'] == cohort:
            grouped[row['participant_id'], row['role']].append(row)
    data, audits = {}, []
    for (subject, role), files in sorted(grouped.items()):
        for row in files:
            entry = ledger[row['release_path']]
            if entry['record_partition'] == 'test':
                raise ValueError('Attempt to access sealed record')
            if role == 'enrollment' and entry['record_partition'] != 'train':
                raise ValueError('Personal enrollment must be explicitly training data')
            if cohort == 'train' and not entry['global_model_fit_allowed']:
                raise ValueError('Support-only records cannot train fusion')
        windows, hardware = baseline.session_features(files)
        keyfile = next(row for row in files if row['modality'] == 'keyboard_csv')
        rows = baseline.read_csv(DATA / cohort / keyfile['release_path'])
        events, differences, unknown = [], [], 0
        for row in rows:
            start, end, duration = [float(row[k]) for k in ['Elapsed Start Time', 'Elapsed Release Time', 'Duration']]
            if not np.isfinite([start, end, duration]).all() or end < start:
                raise ValueError('Unreliable keyboard elapsed timestamps')
            code = mapped_key(row['Key'])
            unknown += code == 0
            differences.append(abs(end - start - duration))
            events.append((start, end, code))
        if unknown / max(len(events), 1) > .01 or np.median(differences) > .01:
            raise ValueError(f'Key mapping/clock gate failed for {subject}/{role}')
        events = np.array(sorted(events, key=lambda e: e[0]))
        sequences, lengths = [], []
        for window in windows:
            start = window['start']
            k = events[(events[:, 0] >= start) & (events[:, 1] < start + 30)]
            if len(k) < 5:
                raise ValueError('Window diverges from baseline acceptance rule')
            hold = k[:-1, 1] - k[:-1, 0]
            ud = k[1:, 0] - k[:-1, 1]
            dd = np.diff(k[:, 0]); uu = np.diff(k[:, 1])
            seq = np.column_stack([k[:-1, 2] / 255, hold, ud, dd, uu])[:50].astype(np.float32)
            sequences.append(seq)
            lengths.append(len(seq))
        if sequences:
            vectors = encoder.embed(sequences)
            for window, vector, length in zip(windows, vectors, lengths):
                window['embedding'] = vector.tolist()
                window['sequence_length'] = length
        audits.append({'subject': subject, 'role': role, 'record_partition': 'train' if role == 'enrollment' else cohort,
                       'events': len(events), 'unknown_codes': unknown,
                       'median_duration_disagreement_seconds': float(np.median(differences)),
                       'windows': len(windows), 'sequence_lengths': lengths})
        data.setdefault(subject, {})[role] = {'windows': windows, 'hardware': hardware}
    valid = {s: d for s, d in data.items() if len(d.get('enrollment', {}).get('windows', [])) >= 2 and d.get('probe', {}).get('windows')}
    return valid, audits


def pairs(data):
    x, y, groups = [], [], []
    profiles = {}
    for owner, records in data.items():
        enrolled = records['enrollment']['windows']
        profile = {'gallery': np.array([w['embedding'] for w in enrolled[:5]])}
        for kind in ['mouse', 'keyboard']:
            a = np.array([w[kind] for w in enrolled])
            center = np.median(a, 0)
            profile[kind] = center, np.maximum(np.mean(abs(a - center), 0), .15)
        profiles[owner] = profile
    for owner, profile in profiles.items():
        for actual, records in data.items():
            for window in records['probe']['windows']:
                distance = np.linalg.norm(profile['gallery'] - np.array(window['embedding']), axis=1).mean()
                row = [float(distance)]
                for kind in ['mouse', 'keyboard']:
                    center, spread = profile[kind]
                    row += np.minimum(abs(np.array(window[kind]) - center) / spread, 8).tolist()
                x.append(row); y.append(int(owner == actual)); groups.append((owner, actual, str(window['start'])))
    return np.array(x), np.array(y), np.array(groups)


def train(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'preregistered.json').open('x') as f:
        json.dump(PROTOCOL, f, indent=2)
    existing = json.loads((baseline.OUT / 'frozen.json').read_text())
    group = existing['protocol']
    data, audit = load('train', Encoder())
    groups = [group[name] for name in ['fit_identities', 'selection_identities', 'calibration_identities']]
    matrices = [pairs({s: data[s] for s in ids}) for ids in groups]
    (xf, yf, _), (xs, ys, _), (xc, yc, _) = matrices
    frozen = {'protocol': PROTOCOL, 'identity_groups': groups, 'train_audit': audit,
              'manifest_sha256': checksum(DATA / 'split_manifest.json'),
              'roles_sha256': checksum(DATA / 'record_roles.json'), 'models': {}, 'candidates': {}}
    for kind, columns in FEATURES.items():
        candidates = []
        for c in PROTOCOL['fusion_candidates_C']:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=20260920))
            model.fit(xf[:, columns], yf)
            score = -model.decision_function(xs[:, columns])
            value = eer(score[ys == 1], score[ys == 0])
            candidates.append((value, c, model))
        value, c, model = min(candidates, key=lambda item: item[0])
        score = -model.decision_function(xc[:, columns])
        thresholds = {name: threshold_for(score[yc == 1], score[yc == 0], target)
                      for name, target in [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
        joblib.dump(model, output / f'{kind}.joblib')
        frozen['models'][kind] = {'columns': columns, 'C': c, 'selection_eer': value,
                                  'thresholds': thresholds, 'model_sha256': checksum(output / f'{kind}.joblib')}
        frozen['candidates'][kind] = [{'C': c0, 'selection_eer': e0} for e0, c0, _ in candidates]
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({'training_selection': frozen['models']}), flush=True)


def validate(output):
    frozen = json.loads((output / 'frozen.json').read_text())
    if frozen['manifest_sha256'] != checksum(DATA / 'split_manifest.json') or frozen['roles_sha256'] != checksum(DATA / 'record_roles.json'):
        raise ValueError('Data roles changed after freezing')
    if frozen['protocol']['source_model_sha256'] != checksum(SOURCE / 'model.pt'):
        raise ValueError('Source encoder changed after freezing')
    data, audit = load('dev', Encoder())
    x, y, groups = pairs(data)
    result = {'protocol': frozen['protocol'], 'subjects': sorted(data), 'dev_audit': audit,
              'evaluation_split': 'dev', 'results': {}, 'frozen_sha256': checksum(output / 'frozen.json')}
    for kind, config in frozen['models'].items():
        if checksum(output / f'{kind}.joblib') != config['model_sha256']:
            raise ValueError('Fusion model changed after freezing')
        model = joblib.load(output / f'{kind}.joblib')
        scores = -model.decision_function(x[:, config['columns']])
        result['results'][kind] = baseline.measure(y, scores, config['thresholds'])
    np.savez_compressed(output / 'dev_pairs.npz', X=x, y=y, groups=groups)
    (output / 'dev_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['results'], indent=2), flush=True)


def self_check():
    assert mapped_key('key.shift_r') == 16 and mapped_key('key.f4') == 115
    assert mapped_key('a') == 65 and mapped_key('\x01') == 65 and mapped_key('<49>') == 49
    assert mapped_key('unrecognized') == 0
    encoder = Encoder()
    sequence = np.zeros((9, 5), dtype=np.float32); sequence[:, 0] = 65 / 255; sequence[:, 1:] = .1
    short = encoder.embed([sequence])[0]
    with torch.no_grad():
        direct = encoder.model(torch.as_tensor(encoder.transform.apply(sequence[None]))).numpy()[0]
    assert np.allclose(short, direct, rtol=0, atol=1e-6)
    try:
        load('test', encoder)
        raise AssertionError('Test access permitted')
    except ValueError:
        pass
    print('Passed key mapping, masked-prefix embedding equivalence and sealed-test checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'dev', 'both', 'self-check'], default='both')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'self-check':
        self_check(); return
    if args.stage in ('train', 'both'):
        train(args.output)
    if args.stage in ('dev', 'both'):
        if (args.output / 'dev_results.json').exists():
            raise SystemExit('Preserve existing development results')
        validate(args.output)


if __name__ == '__main__':
    main()
