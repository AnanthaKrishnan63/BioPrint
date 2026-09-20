"""Predeclared train-only latent scoring comparison, after frozen FCN training."""
import json
import pathlib
import time
import numpy as np
import torch
from pointer_sapimouse_benchmark import (FCN, load, embed, score_profiles, aggregate, DATA, OUT as SOURCE_OUT, stats, threshold_at_far)
from engine.scorer import _fit_arrays, _deviations

OUT = SOURCE_OUT.parent / 'pointer_sapimouse_optimized'


def latent_scores(enroll, labels, probe, method, save=False):
    users = sorted(set(labels))
    scores = []
    for user in users:
        own = enroll[labels == user]
        if method == 'cosine':
            normalized = own / np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-12)
            center = normalized.mean(axis=0)
            center /= max(np.linalg.norm(center), 1e-12)
            score = (probe / np.maximum(np.linalg.norm(probe, axis=1, keepdims=True), 1e-12)) @ center
            if save:
                np.savez_compressed(OUT / f'cosine_{user}.npz', center=center)
        elif method == 'latent_manhattan':
            center, spread = _fit_arrays(own, np.asarray([f'pointer.{i}' for i in range(own.shape[1])]))
            score = -_deviations(center, spread, probe, 6).mean(axis=1)
            if save:
                np.savez_compressed(OUT / f'latent_manhattan_{user}.npz', center=center, spread=spread, cap=6)
        else:
            raise ValueError(method)
        scores.append(score)
    return np.asarray(scores), np.asarray(users)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise SystemExit('Existing optimization results are frozen; do not rerun from dev feedback')
    if not (SOURCE_OUT / 'results.json').exists():
        raise SystemExit('Wait for original fixed training run to finish')
    torch.set_num_threads(2)
    m = json.loads((DATA / 'split_manifest.json').read_text())
    saved = torch.load(SOURCE_OUT / 'fcn_training_best.pt', weights_only=True)
    model = FCN(saved['classes']).eval()
    model.load_state_dict(saved['state_dict'])
    train_files = lambda length: [e for e in m['files'] if e['split'] == 'train' and e['role'] == 'calibration' and e['path'].endswith(f'_{length}min.csv')]
    enrollment, _, labels, _ = load(train_files(3))
    probe, _, probe_labels, probe_groups = load(train_files(1))
    enrollment, probe = embed(model, enrollment), embed(model, probe)
    original = json.loads((SOURCE_OUT / 'frozen_training_selection.json').read_text())
    calibration = dict(original['calibration'])
    thresholds = dict(original['thresholds'])
    for method in ['cosine', 'latent_manhattan']:
        matrix, users = latent_scores(enrollment, labels, probe, method)
        for n in [1, 3, 5]:
            grouped, actual, _ = aggregate(matrix, probe_labels, probe_groups, n)
            y = np.asarray([actual == u for u in users]).ravel().astype(int)
            name = f'{method}@{n}'
            thresholds[name] = {str(f): threshold_at_far(y, grouped.ravel(), f) for f in [.001, .01, .05]}
            calibration[name] = stats(y, grouped.ravel(), thresholds[name]['0.01'])
    selected = min([n for n in calibration if not n.startswith('scaled_manhattan')], key=lambda n: (calibration[n]['frr'], calibration[n]['eer_descriptive']))
    n = int(selected.split('@')[1])
    frozen = {'selected': selected, 'calibration': calibration, 'thresholds': thresholds, 'training_checkpoint_epoch': saved['epoch'], 'plan': json.loads((SOURCE_OUT / 'optimization_plan.json').read_text()), 'input_events_per_decision': 128 * n + 1, 'displacements_per_decision': 128 * n}
    (OUT / 'frozen_training_selection.json').write_text(json.dumps(frozen, indent=2))
    print('FROZEN', json.dumps(frozen), flush=True)
    # No dev measurement or result affects the selection above.
    def dev_files(length):
        role = 'dev_support_enrollment' if length == 3 else 'dev_probe'
        selected = [e for e in m['files'] if e['role'] == role]
        assert all(e['split'] == ('train' if length == 3 else 'dev') and e['path'].endswith(f'_{length}min.csv') for e in selected)
        return selected
    enroll_raw, enroll_hand, enroll_labels, _ = load(dev_files(3))
    probe_raw, probe_hand, probe_labels, probe_groups = load(dev_files(1))
    enroll_deep, probe_deep = embed(model, enroll_raw), embed(model, probe_raw)
    medians = np.load(SOURCE_OUT / 'training_feature_medians.npy')
    fill = lambda x: np.where(np.isnan(x), medians, x)
    reports = {}
    for name in dict.fromkeys([f'scaled_manhattan@{n}', f'ocsvm:0.5:raw@{n}', selected]):
        method = name.split('@')[0]
        if method in ['cosine', 'latent_manhattan']:
            matrix, users = latent_scores(enroll_deep, enroll_labels, probe_deep, method, save=True)
        else:
            a, b = (fill(enroll_hand), fill(probe_hand)) if method == 'scaled_manhattan' else (enroll_deep, probe_deep)
            matrix, users = score_profiles(a, enroll_labels, b, probe_labels, method)
        grouped, actual, sessions = aggregate(matrix, probe_labels, probe_groups, n)
        truth = np.asarray([actual == u for u in users]).ravel().astype(int)
        reports[name] = {}
        for far, threshold in thresholds[name].items():
            r = stats(truth, grouped.ravel(), threshold)
            per = {u: stats((actual == u).astype(int), grouped[i], threshold) for i, u in enumerate(users)}
            r['macro_eer_descriptive'] = float(np.mean([v['eer_descriptive'] for v in per.values()]))
            r['macro_frr'] = float(np.mean([v['frr'] for v in per.values()]))
            r['per_claimed_user'] = per
            reports[name][far] = r
        np.savez_compressed(OUT / f'{name.replace(":", "_")}_dev_scores.npz', scores=grouped, true_user=actual, users=users, sessions=sessions)
    result = {'selected_from_training_only': selected, 'dev': reports, 'counts': {'train_calibration_users': len(set(labels)), 'dev_enrollment_users': len(set(enroll_labels)), 'dev_probe_users': len(set(probe_labels)), 'training_support_enrollment_windows': len(enroll_deep), 'dev_probe_windows': len(probe_deep)}, 'input_events_per_decision': n * 128 + 1, 'displacements_per_decision': n * 128, 'checkpoint_epoch': saved['epoch'], 'test_accessed': False, 'limitations': ['Predeclared scoring variants; all ranking/thresholds use12training-calibration users only.', 'Unseen-encoder dev cohort enrolled with3min TRAINING-support files;1min DEV probes never fit or scale models.', 'Dataset users/devices not a same-device attacker study; no claim of production FAR.', 'No current2025SOTA equivalence or multimodal fusion claim.']}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:{key:value for key,value in v['0.01'].items() if key!='per_claimed_user'} for k,v in reports.items()}, indent=2))

if __name__ == '__main__':
    main()
