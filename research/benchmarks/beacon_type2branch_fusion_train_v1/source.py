"""Fit, select, and calibrate paired fusion using disjoint BEACON TRAIN roles."""
import copy
import json
from pathlib import Path
import numpy as np
from beacon_type2branch_embed import ROOT, PROTOCOL, PROTOCOL_SHA256, verify_hashes
from type2branch_continuity import sha
from beacon_type2branch_selection import selection_key
from eval.strict_cmu import threshold_for, eer, rates

PREP = ROOT / 'research/benchmarks/beacon_type2branch_train_features_v1'
EMBED = ROOT / 'research/benchmarks/beacon_type2branch_train_embeddings_v1'
OUT = ROOT / 'research/benchmarks/beacon_type2branch_fusion_train_v1'


def attach_embeddings(data, arrays, expected_lengths):
    """Attach each embedding once by exact person/role/start, never row position."""
    result = copy.deepcopy(data)
    lookup = {}
    for subject, records in result.items():
        if set(records) != {'enrollment', 'probe'}:
            raise ValueError('Both record roles required')
        for role, record in records.items():
            windows = record['windows']
            if len(windows) < (2 if role == 'enrollment' else 1):
                raise ValueError('Incomplete prescribed gallery/probe support')
            starts = [w['start'] for w in windows]
            if starts != sorted(starts) or not np.isfinite(starts).all():
                raise ValueError('Chronological finite windows required')
            for window in windows:
                key = (subject, role, float(window['start']))
                if key in lookup or 'type2branch_embedding' in window:
                    raise ValueError('Duplicate window or repeated attachment')
                lookup[key] = window
    n = len(lookup)
    embeddings = np.asarray(arrays['embeddings'])
    if embeddings.shape != (n, 256) or not np.isfinite(embeddings).all():
        raise ValueError('Finite one-to-one Nx256 embeddings required')
    values = [np.asarray(arrays[k]) for k in ['subject', 'role', 'start', 'true_length']]
    if any(v.shape != (n,) for v in values) or values[3].dtype.kind not in 'iu':
        raise ValueError('Aligned explicit lengths and window metadata required')
    if set(expected_lengths) != set(lookup):
        raise ValueError('Length audit must cover the exact windows')
    seen = set()
    for embedding, subject, role, start, length in zip(embeddings, *values, strict=True):
        key = (str(subject), str(role), float(start))
        if key not in lookup or key in seen or not 5 <= length <= 100 or expected_lengths[key] != length:
            raise ValueError('Window identity/cardinality/length mismatch')
        seen.add(key)
        lookup[key]['type2branch_embedding'] = embedding.tolist()
        lookup[key]['type2branch_true_length'] = int(length)
    return result


def pairs(data, old_pairs=None):
    if old_pairs is None:
        from beacon_dual_neural_benchmark import pairs as old_pairs
    x, y, groups = old_pairs(data)
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    expected = [(owner, actual, str(w['start'])) for owner in data
                for actual in data for w in data[actual]['probe']['windows']]
    if (x.shape != (len(expected), 34) or y.shape != (len(expected),)
            or groups.shape != (len(expected), 3) or groups.tolist() != [list(k) for k in expected]
            or not np.array_equal(y, [int(a == b) for a, b, _ in expected]) or not np.isfinite(x).all()):
        raise ValueError('Original34-column claim ordering changed')
    galleries = {s: np.asarray([w['type2branch_embedding'] for w in d['enrollment']['windows'][:5]])
                 for s, d in data.items()}
    probes = {(s, str(w['start'])): np.asarray(w['type2branch_embedding'])
              for s, d in data.items() for w in d['probe']['windows']}
    distance = [np.linalg.norm(galleries[owner] - probes[actual, start], axis=1).mean()
                for owner, actual, start in expected]
    output = np.column_stack([x, distance])
    if not np.isfinite(output).all():
        raise ValueError('Finite distances required')
    return output, y, groups


def measure(labels, scores, thresholds):
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    return {'eer_diagnostic': eer(genuine, impostor), 'genuine_comparisons': len(genuine),
            'impostor_comparisons': len(impostor), 'operating_points': {
                name: {**rates(genuine, impostor, threshold),
                       'false_rejections': int((genuine > threshold).sum()),
                       'false_acceptances': int((impostor <= threshold).sum())}
                for name, threshold in thresholds.items()}}


def main():
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(PROTOCOL.read_text()); verify_hashes(protocol)
    reports = {}; plans = {}
    for path, status in [(PREP, 'paired_train_features_complete'), (EMBED, 'paired_train_embeddings_complete')]:
        reports[path] = json.loads((path / 'report.json').read_text())
        plans[path] = json.loads((path / 'plan.json').read_text())
        if reports[path]['status'] != status or plans[path]['protocol_sha256'] != PROTOCOL_SHA256:
            raise ValueError('Completed protocol-bound TRAIN artifact required')
        verify_hashes(plans[path])
    inputs = {str((PREP / 'paired_data.json').relative_to(ROOT)): reports[PREP]['paired_data_sha256'],
              str((EMBED / 'embeddings.npz').relative_to(ROOT)): reports[EMBED]['embeddings_sha256']}
    verify_hashes({'input_sha256': inputs})
    if reports[EMBED]['features_sha256'] != reports[PREP]['features_sha256']:
        raise ValueError('Embedding preparation binding mismatch')
    sources = [Path(__file__), PROTOCOL, ROOT / 'scripts/beacon_type2branch_embed.py',
               ROOT / 'scripts/type2branch_continuity.py', ROOT / 'scripts/beacon_type2branch_selection.py']
    sources += [p / name for p in [PREP, EMBED] for name in ['plan.json', 'report.json']]
    plan = {'protocol_sha256': PROTOCOL_SHA256, 'scope': 'Disjoint TRAIN fit/selection/calibration only',
            'source_sha256': {**protocol['source_sha256'], **{str(p.relative_to(ROOT)): sha(p) for p in sources}},
            'input_sha256': inputs}
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    data = json.loads((PREP / 'paired_data.json').read_text())
    if set(data) != set(sum(protocol['identity_groups'], [])):
        raise ValueError('All prescribed TRAIN identities required')
    lengths = {(r['subject'], r['role'], float(w['start'])): w['true_length']
               for r in reports[PREP]['type2branch_audit'] for w in r['windows']}
    with np.load(EMBED / 'embeddings.npz', allow_pickle=False) as arrays:
        data = attach_embeddings(data, arrays, lengths)
    matrices = [pairs({s: data[s] for s in ids}) for ids in protocol['identity_groups']]
    for name, (x, y, groups) in zip(['fit', 'selection', 'calibration'], matrices):
        np.savez_compressed(OUT / f'{name}_pairs.npz', X=x, y=y, groups=groups)
    (xf, yf, _), (xs, ys, _), (xc, yc, _) = matrices
    frozen = {'protocol': protocol, 'protocol_sha256': PROTOCOL_SHA256,
              'identity_groups': protocol['identity_groups'], 'models': {}, 'candidates': {},
              'train_audit': reports[PREP]['type2branch_audit'], 'dev_test_observations_decoded': False}
    for kind, columns in protocol['models'].items():
        candidates = []
        for c in protocol['candidate_C']:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight='balanced', max_iter=1000, random_state=20260920))
            model.fit(xf[:, columns], yf)
            score = -model.decision_function(xs[:, columns])
            key = selection_key(ys, score, c)
            candidates.append((key, model, score))
        key, model, selection_scores = min(candidates, key=lambda item: item[0])
        calibration_scores = -model.decision_function(xc[:, columns])
        thresholds = {name: threshold_for(calibration_scores[yc == 1], calibration_scores[yc == 0], target)
                      for name, target in [('far_1pct', .01), ('far_5pct', .05), ('eer', None)]}
        joblib.dump(model, OUT / f'{kind}.joblib')
        target = OUT / f'{kind}_train_scores.npz'
        np.savez_compressed(target, selection_labels=ys, selection_scores=selection_scores,
                            candidate_selection_scores=np.stack([v[2] for v in candidates]),
                            candidate_C=np.array(protocol['candidate_C']),
                            calibration_labels=yc, calibration_scores=calibration_scores)
        frozen['models'][kind] = {'columns': columns, 'C': key[2], 'selection_key': list(key),
            'thresholds': thresholds, 'calibration_metrics': measure(yc, calibration_scores, thresholds),
            'model_sha256': sha(OUT / f'{kind}.joblib'), 'train_scores_sha256': sha(target)}
        frozen['candidates'][kind] = [{'C': k[2], 'selection_key': list(k)} for k, _, _ in candidates]
    frozen['pairs_sha256'] = {name: sha(OUT / f'{name}_pairs.npz') for name in ['fit', 'selection', 'calibration']}
    for manifest in [plan, protocol, *plans.values()]: verify_hashes(manifest)
    frozen['status'] = 'paired_train_fusion_complete'
    (OUT / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps({'status': frozen['status'], 'models': frozen['models']}), flush=True)


if __name__ == '__main__':
    main()
