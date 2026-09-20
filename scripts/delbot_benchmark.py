"""DELBOT source-disjoint bot research; reserved GAN/phone/fast sets never read.

This assesses bot-family/device-source transfer, not user identity. Human subject
IDs are not available: folder groups must not be presented as people.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import numpy as np
from device_benchmark import metrics

DATA = ROOT / 'datasets/delbot'
OUT = ROOT / 'research/benchmarks'
GROUPS = {'circles_bot_pynput': 'train', 'circles_bot_naturalmousemotion': 'train',
          'circles_bot_pyhm': 'dev', 'circles_bot_gan': 'test',
          'circles_human_pc1': 'train', 'circles_human_pc2_pad': 'train',
          'circles_human_vm': 'train', 'circles_human_pc2': 'dev',
          'circles_human_tel': 'test', 'circles_human_fast': 'test'}
CALIBRATION = {'circles_bot_naturalmousemotion', 'circles_human_vm'}


def prepare():
    DATA.mkdir(exist_ok=True, parents=True)
    archive = DATA / 'dataset.tar.gz'
    for remote, local in [('python/dataset.tar.gz', 'dataset.tar.gz'), ('LICENSE.md', 'LICENSE.md'), ('README.md', 'README.upstream.md')]:
        path = DATA / local
        if not path.exists():
            with urllib.request.urlopen('https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/' + remote, timeout=60) as response:
                payload = response.read(5000000)
            if len(payload) >= 5000000:
                raise ValueError('Unexpected size')
            path.write_bytes(payload)
    # Verify Git blob hash from authors' metadata, pinned before download.
    raw = archive.read_bytes()
    if hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != 'c01565edf71ef02f9f8b15494612324e544f48c4':
        raise ValueError('Archive source hash mismatch')
    with tarfile.open(archive) as source:
        files = {m.name: {'split': GROUPS[m.name.split('/')[0]], 'group': m.name.split('/')[0], 'bytes': m.size}
                 for m in source if m.isfile()}
    manifest = {'version': 1, 'protocol': 'source/family-disjoint split; device-source folders are NOT known person IDs',
                'test_accessed': False, 'archive_sha256': hashlib.sha256(raw).hexdigest(), 'groups': GROUPS, 'files': files}
    path = DATA / 'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to change frozen source splits')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'split_files': dict(Counter(x['split'] for x in files.values())), 'bytes': len(raw)}))


def stats(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return np.r_[np.quantile(v, [.1, .5, .9]), np.mean(v), np.std(v)] if len(v) else np.zeros(5)


def features(raw):
    rows = []
    for row in csv.reader(io.StringIO(raw.decode())):
        if len(row) != 4:
            continue  # source resolution header; do not use hardware as bot evidence
        try:
            rows.append((float(row[0]), row[1], float(row[2]), float(row[3])))
        except ValueError:
            continue
    moves = np.array([[t, x, y] for t, event, x, y in rows if event == 'Move'], dtype=float).reshape(-1, 3)
    if len(moves) < 8 or not np.isfinite(moves).all():
        return None
    moves = moves[np.argsort(moves[:, 0], kind='stable')]
    moves = moves[np.r_[np.diff(moves[:, 0]) > 0, True]]
    if len(moves) < 8:
        return None
    dt = np.diff(moves[:, 0])
    delta = np.diff(moves[:, 1:], axis=0)
    step = np.linalg.norm(delta, axis=1)
    length, duration = step.sum(), dt.sum()
    if length <= 0 or duration <= 0:
        return None
    # Normalize time and spatial scale; excludes display resolution, IDs, source,
    # absolute position, event.isTrusted (unavailable), and class labels.
    speed = (step / length) / (dt / duration)
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    turn = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    vec = np.r_[stats(dt / np.mean(dt)), stats(step / np.mean(step)), stats(speed), stats(abs(turn)),
                np.linalg.norm(moves[-1, 1:] - moves[0, 1:]) / length,
                np.mean(step == 0), np.mean(abs(turn) > np.pi / 2)]
    # Evaluate only the compatible pointer-rule subgroup. No fabricated probe or
    # password metadata is passed to the complete bot gate.
    from contracts import Sample, Meta, PointerEvent
    from engine import bot
    events = [PointerEvent(t=t, x=x, y=y, type={'Move': 'move', 'Pressed': 'down', 'Released': 'up'}[event])
              for t, event, x, y in rows if event in {'Move', 'Pressed', 'Released'}]
    acc = bot._Acc()
    bot._pointer(Sample(keystrokes=[], pointer=events, meta=Meta(submit_via='click')), acc)
    return vec, acc.result().score


def load(split):
    if split not in {'train', 'dev'}:
        raise ValueError('Reserved test traces must remain sealed')
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    x, y, groups, baseline, excluded = [], [], [], [], []
    content = {}
    with tarfile.open(DATA / 'dataset.tar.gz') as archive:
        # Sequential gzip access; reserved entries are skipped without reading.
        for member in archive:
            metadata = manifest['files'].get(member.name)
            if member.isfile() and metadata and metadata['split'] == split:
                content[member.name] = archive.extractfile(member).read()
        for name, metadata in sorted(manifest['files'].items()):
            if metadata['split'] != split:
                continue  # no member content is opened for reserved traces
            output = features(content[name])
            if output is None:
                excluded.append(name)
                continue
            vector, score = output
            x.append(vector)
            y.append(int('_human_' in metadata['group']))
            groups.append(metadata['group'])
            baseline.append(-score)  # larger values mean more human
    return np.asarray(x), np.asarray(y), np.asarray(groups), np.asarray(baseline), excluded


def evaluate():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y, groups, baseline, excluded = load('train')
    calibration = np.isin(groups, list(CALIBRATION))
    models = {
        'geometric_random_forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight='balanced', random_state=20260920, n_jobs=2),
        'geometric_logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(C=.1,
                class_weight='balanced', max_iter=1000, random_state=20260920)),
    }
    calibration_scores = {}
    for name, model in models.items():
        model.fit(x[~calibration], y[~calibration])
        calibration_scores[name] = model.predict_proba(x[calibration])[:, 1]
    # All methods/hyperparameters frozen before this dev load; no tuning on dev.
    xd, yd, gd, bd, excluded_dev = load('dev')
    result = {'version': 1, 'dataset': 'DELBOT-Mouse', 'test_accessed': False,
              'fit_groups': sorted(set(groups[~calibration])), 'calibration_groups': sorted(CALIBRATION),
              'dev_groups': sorted(set(gd)), 'train_excluded': excluded, 'dev_excluded': excluded_dev,
              'train_valid_counts': dict(Counter(groups)), 'dev_valid_counts': dict(Counter(gd)),
              'limitations': ['Device-source folders are not known human identities.',
                'Dev bot family pyhm is held out; GAN reserved test never read.',
                'Geometry baseline, not DELBOT deep model or BeCAPTCHA neuromotor reproduction.',
                'The live baseline covers pointer rules only; no actual login/password/browser environment is available.',
                'Different source acquisition conditions can confound results.'], 'methods': {}}
    for name, model in models.items():
        result['methods'][name] = metrics(yd, model.predict_proba(xd)[:, 1], y[calibration], calibration_scores[name])
    result['methods']['bioprint_pointer_rules_only'] = metrics(yd, bd, y[calibration], baseline[calibration])
    from device_benchmark import rates
    from engine import bot
    result['methods']['bioprint_pointer_rules_only']['original_threshold_dev'] = rates(yd, bd, -bot.THRESHOLD)
    artifact_dir = OUT / 'delbot'
    artifact_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, artifact_dir / f'{name}.joblib')
    names = [f'{family}_{stat}' for family in ['relative_dt', 'relative_step', 'normalized_speed', 'absolute_turn']
             for stat in ['p10', 'median', 'p90', 'mean', 'std']] + ['path_efficiency', 'stationary_fraction', 'reversal_fraction']
    (artifact_dir / 'schema.json').write_text(json.dumps({
        'features': names, 'feature_count': len(names), 'score_direction': 'higher_is_human',
        'input': 'single trajectory geometry vector',
        'models': {name: {'artifact': name + '.joblib', 'operating_points': result['methods'][name]['operating_points']} for name in models},
    }, indent=2) + '\n')
    np.savez_compressed(artifact_dir / 'dev_features.npz', X=xd, y=yd, groups=gd)
    (OUT / 'delbot_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate}[args.command]()
