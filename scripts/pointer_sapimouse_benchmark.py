"""Author-architecture SapiMouse FCN+OCSVM, with sealed-test-safe protocol.

PyTorch port of Antal/Fejer/Buza 2021 FCN architecture. Published evaluation's
probe-dependent score normalization is deliberately replaced by train-only
threshold calibration, and whole-user train/dev/test splits precede all reads.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import pathlib
import time
import zlib
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from pointer_benchmark import features, stats, threshold_at_far
from engine.scorer import _fit_arrays, _deviations

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/benchmarks/sapimouse'
OUT = ROOT / 'research/benchmarks/pointer_sapimouse'

class FCN(nn.Module):
    def __init__(self, classes=60):
        super().__init__()
        layers = []
        channels = 2
        for width, kernel in [(128, 8), (256, 5), (128, 3)]:
            layers.extend([nn.Conv1d(channels, width, kernel, padding='same'), nn.BatchNorm1d(width, eps=1e-3, momentum=.01), nn.ReLU()])
            channels = width
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, classes)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def embed(self, x):
        return self.layers(x).mean(dim=2)

    def forward(self, x):
        return self.classifier(self.embed(x))


def load(entries):
    blocks, handcrafted, labels, sessions = [], [], [], []
    for entry in entries:
        if entry['split'] == 'test_sealed':
            raise ValueError('Refusing test content')
        content = (DATA / entry['path']).read_bytes()
        if zlib.crc32(content) != entry['crc']:
            raise ValueError('Source CRC mismatch')
        rows = list(csv.DictReader(content.decode().splitlines()))
        values = np.asarray([[float(row['client timestamp']) / 1000, float(row['x']), float(row['y'])] for row in rows])
        deltas = np.abs(np.diff(values[:, 1:], axis=0))
        for index in range(0, len(deltas) - 127, 128):
            block = deltas[index:index + 128]
            # Exact row-level z-score intent of the authors: both channels together.
            sd = float(block.std())
            norm = (block - block.mean()) / (sd if sd else .0001)
            vector = features(values[index:index + 128])
            blocks.append(norm.T)
            handcrafted.append(vector if vector is not None else [np.nan] * 29)
            labels.append(entry['path'].split('/')[1])
            sessions.append(entry['path'])
    return np.asarray(blocks, dtype=np.float32), np.asarray(handcrafted), np.asarray(labels), np.asarray(sessions)


def embed(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.embed(torch.from_numpy(x[i:i + 128])).numpy() for i in range(0, len(x), 128)])


def train_model(x, labels, heldout, heldout_labels, epochs):
    users = sorted(str(u) for u in set(labels))
    encode = {u: i for i, u in enumerate(users)}
    y = torch.tensor([encode[u] for u in labels], dtype=torch.long)
    hy = torch.tensor([encode[u] for u in heldout_labels], dtype=torch.long)
    model = FCN(len(users))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=50, min_lr=.0001)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), y), batch_size=16, shuffle=True)
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0., 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(bx)
            count += len(bx)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(torch.from_numpy(heldout[i:i + 128])) for i in range(0, len(heldout), 128)])
            validation_loss = float(nn.functional.cross_entropy(logits, hy))
            accuracy = float((logits.argmax(1) == hy).float().mean())
        scheduler.step(validation_loss)
        record = {'epoch': epoch, 'training_loss': total / count, 'training_session_holdout_loss': validation_loss, 'training_session_holdout_accuracy': accuracy, 'elapsed_seconds': time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({'state_dict': best_state, 'classes': len(users), 'epoch': epoch, 'users': users}, OUT / 'fcn_training_best.pt')
        (OUT / 'training_history.json').write_text(json.dumps(history, indent=2))
    model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_training_session_holdout_loss': best_loss, 'epochs': epochs, 'parameters': sum(p.numel() for p in model.parameters()), 'training_users': users}


def score_profiles(enroll, enroll_labels, probe, probe_labels, mode, save=False):
    users = sorted(set(enroll_labels))
    matrix = []
    for user in users:
        own = enroll[enroll_labels == user]
        if mode == 'scaled_manhattan':
            center, spread = _fit_arrays(own, np.asarray([f'pointer.{i}' for i in range(own.shape[1])]))
            score = -_deviations(center, spread, probe, 6).mean(axis=1)
            if save:
                np.savez_compressed(OUT / f'{mode}_{user}.npz', center=center, spread=spread, cap=6)
        else:
            _, nu_string, normalization = mode.split(':')
            nu = float(nu_string)
            classifier = OneClassSVM(gamma='scale', nu=nu).fit(own)
            score = classifier.score_samples(probe)
            if normalization == 'enroll':
                # Correct scale differs with profile sample count; denominator uses enrollment only.
                score = score / (nu * len(own))
            if save:
                joblib.dump(classifier, OUT / f'{mode.replace(":", "_")}_{user}.joblib')
        matrix.append(score)
    return np.asarray(matrix), np.asarray(users)


def aggregate(matrix, labels, groups, n):
    # Never average across users/sessions; use nonoverlapping blocks only.
    columns, out_labels, out_groups = [], [], []
    for group in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        for start in range(0, len(indices) - n + 1, n):
            block = indices[start:start + n]
            columns.append(matrix[:, block].mean(axis=1))
            out_labels.append(labels[block[0]])
            out_groups.append(group)
    return np.asarray(columns).T, np.asarray(out_labels), np.asarray(out_groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume-trained', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise SystemExit('Frozen dev results already exist; do not re-optimize from dev.')
    torch.set_num_threads(2)
    torch.manual_seed(11235)
    np.random.seed(11235)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    entries = manifest['files']
    select = lambda role, length: [e for e in entries if e['split'] == 'train' and e['role'] == role and e['path'].endswith(f'_{length}min.csv')]
    fit, fit_hand, fit_labels, _ = load(select('representation', 3))
    heldout, _, heldout_labels, _ = load(select('representation', 1))
    print(json.dumps({'fit_windows': len(fit), 'training_session_holdout_windows': len(heldout)}), flush=True)
    if args.resume_trained:
        saved = torch.load(OUT / 'fcn_training_best.pt', weights_only=True)
        model = FCN(saved['classes'])
        model.load_state_dict(saved['state_dict'])
        training = {'best_epoch': saved['epoch'], 'resumed': True, 'training_users': saved['users']}
    else:
        model, training = train_model(fit, fit_labels, heldout, heldout_labels, args.epochs)
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    entries = manifest['files']
    enrollment, enrollment_hand, enrollment_labels, _ = load(select('calibration', 3))
    probes, probe_hand, probe_labels, probe_groups = load(select('calibration', 1))
    enrollment_deep, probe_deep = embed(model, enrollment), embed(model, probes)
    medians = np.nanmedian(fit_hand, axis=0)
    fill = lambda a: np.where(np.isnan(a), medians, a)
    enrollment_hand, probe_hand = fill(enrollment_hand), fill(probe_hand)
    np.save(OUT / 'training_feature_medians.npy', medians)
    methods = ['scaled_manhattan', 'ocsvm:0.5:raw', 'ocsvm:0.5:enroll', 'ocsvm:0.1:enroll']
    calibration, thresholds = {}, {}
    for method in methods:
        a, b = (enrollment_hand, probe_hand) if method == 'scaled_manhattan' else (enrollment_deep, probe_deep)
        matrix, users = score_profiles(a, enrollment_labels, b, probe_labels, method)
        for n in [1, 3, 5]:
            grouped, grouped_labels, _ = aggregate(matrix, probe_labels, probe_groups, n)
            y = np.asarray([grouped_labels == u for u in users]).ravel().astype(int)
            s = grouped.ravel()
            name = f'{method}@{n}'
            thresholds[name] = {str(f): threshold_at_far(y, s, f) for f in [.001, .01, .05]}
            calibration[name] = stats(y, s, thresholds[name]['0.01'])
    selected = min([n for n in calibration if n.startswith('ocsvm')], key=lambda n: (calibration[n]['frr'], calibration[n]['eer_descriptive']))
    block_count = int(selected.split('@')[1])
    baseline = f'scaled_manhattan@{block_count}'
    author = f'ocsvm:0.5:raw@{block_count}'
    frozen = {'selected': selected, 'baseline': baseline, 'author_architecture_default': author, 'calibration': calibration, 'thresholds': thresholds, 'training': training, 'protocol': manifest['protocol']}
    (OUT / 'frozen_training_selection.json').write_text(json.dumps(frozen, indent=2))
    print('FROZEN', json.dumps(frozen), flush=True)
    # First access to dev measurements follows frozen hyperparameters and thresholds.
    def dev_select(length):
        role = 'dev_support_enrollment' if length == 3 else 'dev_probe'
        selected = [e for e in entries if e['role'] == role]
        assert all(e['split'] == ('train' if length == 3 else 'dev') and e['path'].endswith(f'_{length}min.csv') for e in selected)
        return selected
    dev_enroll, dev_enroll_hand, dev_enroll_labels, _ = load(dev_select(3))
    dev_probe, dev_probe_hand, dev_probe_labels, dev_probe_groups = load(dev_select(1))
    dev_enroll_deep, dev_probe_deep = embed(model, dev_enroll), embed(model, dev_probe)
    reports = {}
    for name in dict.fromkeys([baseline, author, selected]):
        method, n = name.split('@')
        a, b = (fill(dev_enroll_hand), fill(dev_probe_hand)) if method == 'scaled_manhattan' else (dev_enroll_deep, dev_probe_deep)
        matrix, users = score_profiles(a, dev_enroll_labels, b, dev_probe_labels, method, save=True)
        grouped, grouped_labels, grouped_sessions = aggregate(matrix, dev_probe_labels, dev_probe_groups, int(n))
        y = np.asarray([grouped_labels == u for u in users]).ravel().astype(int)
        reports[name] = {}
        for far, threshold in thresholds[name].items():
            metrics = stats(y, grouped.ravel(), threshold)
            per_user = {u: stats((grouped_labels == u).astype(int), grouped[i], threshold) for i, u in enumerate(users)}
            metrics['macro_eer_descriptive'] = float(np.mean([v['eer_descriptive'] for v in per_user.values()]))
            metrics['macro_frr'] = float(np.mean([v['frr'] for v in per_user.values()]))
            metrics['per_claimed_user'] = per_user
            reports[name][far] = metrics
        np.savez_compressed(OUT / f'{name.replace(":", "_")}_dev_scores.npz', scores=grouped, true_user=grouped_labels, users=users, sessions=grouped_sessions)
    result = {'dataset': 'SapiMouse 2020', 'training': training, 'selected_from_training_only': selected, 'dev': reports, 'counts': {'representation_users': 60, 'training_calibration_users': 12, 'dev_users': 24, 'sealed_test_users': 24, 'fit_windows': len(fit), 'training_session_holdout_windows': len(heldout), 'calibration_enroll_windows': len(enrollment), 'calibration_probe_windows': len(probes), 'training_support_enrollment_windows': len(dev_enroll), 'dev_probe_windows': len(dev_probe)}, 'limitations': ['Author FCN architecture and row z-score reproduced in PyTorch; training protocol and optimizer implementation differ.', 'Encoder-training users are separate from dev cohort;3min TRAINING support enrolls profiles;1min DEV probes never fit models.', 'Source archive contains sealed subjects but their contents were never extracted or read.', 'Device/DPI/task differences may aid recognition; no same-device attacker verification.', 'No test evaluation; EER descriptive on dev, deployed thresholds from training calibration.', 'Score averaging restricted to nonoverlapping within-session blocks, unlike source implementation.', 'Not a claim to reproduce current 2025 SOTA ResNet+GRU.']}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:{f:{m:v for m,v in r.items() if m != 'per_claimed_user'} for f,r in results.items()} for k,results in reports.items()}, indent=2))

if __name__ == '__main__':
    main()
