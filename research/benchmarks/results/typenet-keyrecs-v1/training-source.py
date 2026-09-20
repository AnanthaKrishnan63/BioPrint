"""TypeNet architecture transfer to KeyRecs; NOT published SOTA reproduction.

Source architecture: https://arxiv.org/html/2101.05570 sections IV-A--D.
The 200,458-parameter network is reproduced, including gatewise recurrent
dropout. Data, training budget, clipping, and cohort protocol differ explicitly.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_keystrokes import DATA, prepare, checksum
from keystroke_benchmark import blocks, flatten, calibrated_threshold, curve_metrics, evaluate

OUT = ROOT / 'research/benchmarks/results/typenet-keyrecs-v1'
KEYS = {'Backspace': 8, 'Tab': 9, 'Enter': 13, 'Shift': 16, 'Control': 17,
        'Alt': 18, 'Pause': 19, 'CapsLock': 20, 'Escape': 27, 'Space': 32,
        ' ': 32, 'PageUp': 33, 'PageDown': 34, 'End': 35, 'Home': 36,
        'ArrowLeft': 37, 'ArrowUp': 38, 'ArrowRight': 39, 'ArrowDown': 40,
        'Insert': 45, 'Delete': 46, 'Meta': 91, 'AltGraph': 225}
PROTOCOL = {
    'architecture': 'input BN5, LSTM128, BN128, dropout0.5, LSTM128, last hidden embedding128',
    'recurrent_dropout': .2, 'recurrent_mask': 'four gate-specific masks fixed across sequence time',
    'expected_trainable_parameters': 200458,
    'source': 'https://arxiv.org/html/2101.05570', 'source_sections': 'IV-A through IV-D',
    'loss': 'max(0, squared_distance(anchor,positive)-squared_distance(anchor,negative)+1.5)',
    'optimizer': 'Adam beta1=.9 beta2=.999 epsilon=1e-8',
    'learning_rates': [.05, .001], 'max_epochs': 20, 'batches_per_epoch': 20,
    'triplets_per_batch': 32, 'early_stop_patience': 5, 'seed': 20260920,
    'gallery_sequences_per_subject': 5,
    'input': ['key1 mapped to ASCII-like code /255', 'DU.key1.key1', 'UD.key1.key2', 'DD.key1.key2', 'UU.key1.key2'],
    'sequence_length': 50, 'window_overlap': 0,
    'selection': 'S1 chronological first50% fit, next25% select, last25% calibrate; no S2 tuning',
    'deviations': ['KeyRecs rather than Aalto; only active79 identities, no unseen-user generalization claim',
       'KeyRecs provides key labels rather than original numeric event.keyCode; documented mapping, unknown0',
       '50 full digraph rows per sequence; final incomplete windows dropped rather than zero padded',
       'Fit-only timing imputation and .5/99.5 percentile clipping for source outliers',
       'CPU budget20epochs20batches32triplets differs from200epochs150batches512sequences',
       'Second LR=.001 is predeclared small-data adaptation; source LR=.05 included',
       'Five S1 gallery samples per account; same accounts in S2, no open-set claim'],
    'test_accessed': False,
}


def keycode(label):
    if label in KEYS:
        return KEYS[label]
    if len(label) == 1:
        value = ord(label.upper()) if label.isascii() and label.isalpha() else ord(label)
        return value if value <= 255 else 0
    return 0


def sequences(split):
    if split not in ('train', 'dev'):
        raise ValueError('Test is sealed')
    manifest = prepare()
    active, session = set(manifest['active_subjects']), '1' if split == 'train' else '2'
    grouped, audit = defaultdict(list), Counter()
    with (DATA / 'free-text.csv').open() as source:
        next(source)
        for line in source:
            participant, row_session, opaque = line.split(',', 2)
            if participant not in active or row_session != session:
                continue
            suffix = opaque.rstrip('\r\n')
            if suffix.endswith(','):
                suffix = suffix[:-1]
            prefix, *timing = suffix.rsplit(',', 5)
            try:
                keys = next(csv.reader([prefix], strict=True))
                label = keys[0] if len(keys) == 2 else ''
            except csv.Error:
                label = ''
                audit['malformed_key_prefix'] += 1
            code = keycode(label)
            audit['unknown_keycodes'] += code == 0
            hold, dd, du, ud, uu = [float(v) if v else np.nan for v in timing]
            grouped[participant].append([code / 255., hold, ud, dd, uu])
    result = {}
    for subject, values in grouped.items():
        n = len(values) // 50
        result[subject] = np.asarray(values[:n * 50], dtype=np.float32).reshape(n, 50, 5)
        audit['discarded_tail_digraphs'] += len(values) - n * 50
    return result, dict(audit)


class RecurrentDropoutLSTM(nn.Module):
    """Keras-shaped LSTM with genuine recurrent (not inter-layer) dropout."""
    def __init__(self, inputs, hidden=128, recurrent_dropout=.2):
        super().__init__()
        self.hidden, self.recurrent_dropout = hidden, recurrent_dropout
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden, inputs))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(4 * hidden))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh.T)
        with torch.no_grad():
            self.bias[hidden:2 * hidden].fill_(1.)

    def forward(self, x):
        batch, length, _ = x.shape
        h = x.new_zeros(batch, self.hidden)
        c = x.new_zeros(batch, self.hidden)
        projected = F.linear(x, self.weight_ih, self.bias)
        outputs = []
        masks = None
        if self.training and self.recurrent_dropout:
            masks = F.dropout(x.new_ones(4, batch, self.hidden), self.recurrent_dropout, training=True)
        weights = self.weight_hh.reshape(4, self.hidden, self.hidden).transpose(1, 2)
        for step in range(length):
            if masks is not None:
                recurrent = torch.bmm(h[None] * masks, weights).permute(1, 0, 2).reshape(batch, -1)
            else:
                recurrent = F.linear(h, self.weight_hh)
            i, f, g, o = (projected[:, step] + recurrent).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class TypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Keras BN momentum .99 corresponds to torch update coefficient .01.
        self.input_bn = nn.BatchNorm1d(5, eps=.001, momentum=.01)
        self.lstm1 = RecurrentDropoutLSTM(5)
        self.middle_bn = nn.BatchNorm1d(128, eps=.001, momentum=.01)
        self.dropout = nn.Dropout(.5)
        self.lstm2 = RecurrentDropoutLSTM(128)

    def forward(self, x):
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.lstm1(x)
        x = self.middle_bn(x.transpose(1, 2)).transpose(1, 2)
        return self.lstm2(self.dropout(x))[:, -1]


class TimingTransform:
    def fit(self, x):
        a = x[:, :, 1:].reshape(-1, 4).copy()
        a[~np.isfinite(a)] = np.nan
        self.median = np.nanmedian(a, 0)
        a = np.where(np.isfinite(a), a, self.median)
        self.lo, self.hi = np.quantile(a, [.005, .995], axis=0)
        return self

    def apply(self, x):
        x = x.copy()
        a = x[:, :, 1:]
        x[:, :, 1:] = np.clip(np.where(np.isfinite(a), a, self.median), self.lo, self.hi)
        return x


def embeddings(model, x, batch=256):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            result.append(model(torch.as_tensor(x[start:start + batch])).numpy())
    return np.concatenate(result)


def gallery_scores(query, gallery):
    # gallery shape [accounts, 5, embedding128]. Smaller distance = genuine.
    g = gallery.reshape(-1, gallery.shape[-1]).astype(np.float64)
    q = query.astype(np.float64)
    d2 = np.maximum((q * q).sum(1)[:, None] + (g * g).sum(1)[None, :] - 2 * q @ g.T, 0)
    return -np.sqrt(d2).reshape(len(q), len(gallery), gallery.shape[1]).mean(2)


def selection_eer(scores, y, subjects):
    return float(np.mean([curve_metrics(scores[y == k, k], scores[y != k, k])['eer']
                          for k in range(len(subjects))]))


def train(output):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(PROTOCOL['seed'])
    np.random.seed(PROTOCOL['seed'])
    output.mkdir(parents=True, exist_ok=True)
    registered = output / 'preregistered.json'
    if registered.exists():
        if json.loads(registered.read_text()) != PROTOCOL or (output / 'training-history.jsonl').exists():
            raise ValueError('Existing experiment must not be overwritten')
        # A preprocessing failure before the first training step may be resumed
        # with the identical protocol; no model/selection work is repeated.
    else:
        with registered.open('x') as f:
            json.dump(PROTOCOL, f, indent=2)
    data, audit = sequences('train')
    fit, selection, calibration = blocks(data)
    subjects = sorted(fit)
    x, y = flatten(fit, subjects)
    sx, sy = flatten(selection, subjects)
    cx, cy = flatten(calibration, subjects)
    transform = TimingTransform().fit(x)
    x, sx, cx = [transform.apply(a) for a in (x, sx, cx)]
    gallery_x = np.concatenate([transform.apply(fit[s][:5]) for s in subjects])
    indices = [np.flatnonzero(y == k) for k in range(len(subjects))]
    rng = np.random.default_rng(PROTOCOL['seed'])
    history, best_overall = [], None
    train_start = time.perf_counter()
    for lr in PROTOCOL['learning_rates']:
        torch.manual_seed(PROTOCOL['seed'])
        model = TypeNet()
        assert sum(p.numel() for p in model.parameters()) == 200458
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
        best_eer, best_state, stale = np.inf, None, 0
        for epoch in range(1, PROTOCOL['max_epochs'] + 1):
            started, losses = time.perf_counter(), []
            model.train()
            for batch in range(PROTOCOL['batches_per_epoch']):
                owners = rng.integers(0, len(subjects), PROTOCOL['triplets_per_batch'])
                impostors = (owners + rng.integers(1, len(subjects), len(owners))) % len(subjects)
                pairs = np.array([rng.choice(indices[k], 2, replace=False) for k in owners])
                neg = np.array([rng.choice(indices[k]) for k in impostors])
                picked = np.r_[pairs[:, 0], pairs[:, 1], neg]
                z = model(torch.as_tensor(x[picked]))
                anchor, positive, negative = z.chunk(3)
                loss = F.relu(((anchor - positive) ** 2).sum(1) - ((anchor - negative) ** 2).sum(1) + 1.5).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            gallery = embeddings(model, gallery_x).reshape(len(subjects), 5, 128)
            eer = selection_eer(gallery_scores(embeddings(model, sx), gallery), sy, subjects)
            row = {'lr': lr, 'epoch': epoch, 'loss': float(np.mean(losses)),
                   'training_selection_eer': eer, 'seconds': time.perf_counter() - started}
            history.append(row)
            with (output / 'training-history.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if eer < best_eer - 1e-6:
                best_eer, best_state, stale = eer, copy.deepcopy(model.state_dict()), 0
                if best_overall is None or eer < best_overall['eer']:
                    best_overall = {'eer': eer, 'state': best_state, 'lr': lr, 'epoch': epoch}
                    torch.save(best_state, output / 'best-training-state.pt')
            else:
                stale += 1
            if stale >= PROTOCOL['early_stop_patience']:
                break
    model = TypeNet()
    model.load_state_dict(best_overall['state'])
    torch.save(model.state_dict(), output / 'model.pt')
    gallery = embeddings(model, gallery_x).reshape(len(subjects), 5, 128)
    cscore = gallery_scores(embeddings(model, cx), gallery)
    thresholds = np.array([calibrated_threshold(cscore[cy != k, k]) for k in range(len(subjects))])
    np.savez_compressed(output / 'frozen-arrays.npz', gallery=gallery, thresholds=thresholds,
                        median=transform.median, lo=transform.lo, hi=transform.hi)
    config = {'protocol': PROTOCOL, 'subjects': subjects, 'training_audit': audit,
              'selected_lr': best_overall['lr'], 'selected_epoch': best_overall['epoch'],
              'training_selection_eer': best_overall['eer'], 'parameters': 200458,
              'fit_sequences': len(x), 'selection_sequences': len(sx), 'calibration_sequences': len(cx),
              'training_seconds': time.perf_counter() - train_start,
              'model_sha256': checksum(output / 'model.pt'),
              'split_manifest_sha256': checksum(DATA / 'split-manifest.json'),
              'script_sha256': checksum(Path(__file__)),
              'calibration': evaluate(cscore, cy, subjects, thresholds)}
    (output / 'frozen.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'training-source.py').write_text(Path(__file__).read_text())


def validate(output):
    torch.set_num_threads(2)
    frozen = json.loads((output / 'frozen.json').read_text())
    if frozen['model_sha256'] != checksum(output / 'model.pt'):
        raise ValueError('Model changed after freeze')
    if frozen['split_manifest_sha256'] != checksum(DATA / 'split-manifest.json'):
        raise ValueError('Split changed after freeze')
    model = TypeNet()
    model.load_state_dict(torch.load(output / 'model.pt', weights_only=True))
    with np.load(output / 'frozen-arrays.npz', allow_pickle=False) as arrays:
        transform = TimingTransform()
        transform.median, transform.lo, transform.hi = arrays['median'], arrays['lo'], arrays['hi']
        gallery, thresholds = arrays['gallery'], arrays['thresholds']
    data, audit = sequences('dev')
    subjects = frozen['subjects']
    x, y = flatten(data, subjects)
    scores = gallery_scores(embeddings(model, transform.apply(x)), gallery)
    result = {'evaluation_split': 'dev', 'test_accessed': False,
              'validation': evaluate(scores, y, subjects, thresholds), 'dev_sequences': len(x),
              'dev_audit': audit, 'no_complete_dev_window': [s for s in subjects if len(data[s]) == 0],
              'frozen_sha256': checksum(output / 'frozen.json'),
              'claim': 'TypeNet architecture reproduced, small-data training transfer; NOT published SOTA performance reproduction'}
    (output / 'dev-results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output / 'dev-scores.npz', scores=scores, y=y, subjects=np.array(subjects), thresholds=thresholds)
    print(json.dumps(result['validation']['aggregate_macro_user']), flush=True)


def self_check():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = TypeNet()
    assert sum(p.numel() for p in model.parameters()) == 200458
    x = torch.randn(3, 50, 5)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))
    z = model(x)
    z.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert z.shape == (3, 128)
    custom = RecurrentDropoutLSTM(5, hidden=7, recurrent_dropout=0).eval()
    reference = nn.LSTM(5, 7, batch_first=True)
    with torch.no_grad():
        reference.weight_ih_l0.copy_(custom.weight_ih)
        reference.weight_hh_l0.copy_(custom.weight_hh)
        reference.bias_ih_l0.copy_(custom.bias)
        reference.bias_hh_l0.zero_()
    synthetic = torch.randn(4, 8, 5)
    expected, _ = reference(synthetic)
    assert torch.allclose(custom(synthetic), expected, atol=1e-6, rtol=1e-6)
    try:
        sequences('test')
        raise AssertionError('test not sealed')
    except ValueError:
        pass
    print('Passed TypeNet parameter count, embedding shape, dropout behavior, gradient and sealed-test checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'dev', 'both', 'self-check'], default='both')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'self-check':
        self_check()
        return
    if args.stage in ('train', 'both'):
        train(args.output)
    if args.stage in ('dev', 'both'):
        if (args.output / 'dev-results.json').exists():
            raise SystemExit('Preserve existing dev results; no repeat dev tuning')
        validate(args.output)


if __name__ == '__main__':
    main()
