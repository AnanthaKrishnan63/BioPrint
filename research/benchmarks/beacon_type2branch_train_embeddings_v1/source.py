"""Frozen Type2Branch inference on preregistered paired BEACON TRAIN windows."""
import json
import os
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.research-type2branch-deps'))
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_reference_smoke import module

PROTOCOL = ROOT / 'research/benchmarks/beacon_type2branch_protocol_v1/protocol.json'
PROTOCOL_SHA256 = 'b7c3abc32abb7112975251b3f43e7d0130aefa4e03b2e1dee3ad0ecb0ff5883a'
PREP = ROOT / 'research/benchmarks/beacon_type2branch_train_features_v1'
OUT = ROOT / 'research/benchmarks/beacon_type2branch_train_embeddings_v1'


def verify_hashes(manifest):
    for group in ['source_sha256', 'input_sha256']:
        for name, expected in manifest.get(group, {}).items():
            if sha(ROOT / name) != expected:
                raise ValueError(f'Frozen dependency changed: {name}')


def validate_arrays(arrays, identities):
    x = np.asarray(arrays['features'])
    n = len(x)
    metadata = {key: np.asarray(arrays[key]) for key in ['true_length', 'subject', 'role', 'start']}
    if n == 0 or x.shape != (n, 100, 5) or x.dtype.kind != 'f' or not np.isfinite(x).all():
        raise ValueError('Finite nonempty Nx100x5 features required')
    if any(value.shape != (n,) for value in metadata.values()):
        raise ValueError('Aligned one-dimensional metadata required')
    lengths, subjects, roles, starts = [metadata[k] for k in ['true_length', 'subject', 'role', 'start']]
    if lengths.dtype.kind not in 'iu' or ((lengths < 5) | (lengths > 100)).any():
        raise ValueError('Real lengths5..100 required')
    if subjects.dtype.kind != 'U' or roles.dtype.kind != 'U':
        raise ValueError('Unicode subject and role metadata required')
    if set(subjects) != set(identities) or set(roles) != {'enrollment', 'probe'}:
        raise ValueError('Complete TRAIN cohort and roles required')
    if starts.dtype.kind not in 'fi' or not np.isfinite(starts).all():
        raise ValueError('Finite window starts required')
    ordering = [(str(s), 0 if r == 'enrollment' else 1, float(t)) for s, r, t in zip(subjects, roles, starts)]
    if len(set(ordering)) != n or ordering != sorted(ordering):
        raise ValueError('Unique sorted cohort/role/window ordering required')
    for subject in identities:
        if set(roles[subjects == subject]) != {'enrollment', 'probe'}:
            raise ValueError('Both roles required for every TRAIN identity')
    for features, length in zip(x, lengths):
        if np.any(features[int(length):] != 0):
            raise ValueError('Padding must follow real events and remain zero')
    converted = x.astype('float32')
    if not np.isfinite(converted).all():
        raise ValueError('Features exceed inference precision')
    return converted, metadata


def main():
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Preregistered protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    verify_hashes(protocol)
    prepared = json.loads((PREP / 'report.json').read_text())
    prep_plan = json.loads((PREP / 'plan.json').read_text())
    if prepared['status'] != 'paired_train_features_complete':
        raise ValueError('Completed TRAIN preparation required')
    verify_hashes(prep_plan)
    if (prep_plan.get('source_sha256', {}).get(str(PROTOCOL.relative_to(ROOT))) != PROTOCOL_SHA256
            and prep_plan.get('protocol_sha256') != PROTOCOL_SHA256):
        raise ValueError('Preparation must bind the frozen protocol')
    if sha(PREP / 'features.npz') != prepared['features_sha256']:
        raise ValueError('Prepared features changed')
    pair_path = ROOT / 'research/benchmarks/type2branch_length_pair_v1/report.json'
    pair = json.loads(pair_path.read_text())
    chosen = pair['selected_checkpoint']
    if (pair['status'] != 'paired_train_comparison_complete' or pair['selected_arm'] != 'mixed'
            or chosen['updates'] != 500 or chosen['epoch'] != 4):
        raise ValueError('Expected frozen mixed epoch4/500-update checkpoint')
    checkpoint = Path(chosen['checkpoint'])
    pieces = sorted(checkpoint.parent.glob(checkpoint.name + '.*'))
    if {p.name for p in pieces} != {checkpoint.name + '.index', checkpoint.name + '.data-00000-of-00001'}:
        raise ValueError('Complete single-shard checkpoint required')
    if any(protocol['source_sha256'].get(str(p.relative_to(ROOT))) != sha(p) for p in pieces):
        raise ValueError('Checkpoint must be pinned by protocol')
    original_path = ROOT / 'research/benchmarks/type2branch_train_v1/plan.json'
    original = json.loads(original_path.read_text())
    verify_hashes(original)
    sources = [Path(__file__), PROTOCOL, PREP / 'plan.json', PREP / 'report.json', pair_path,
               original_path, ROOT / 'scripts/type2branch_continuity.py',
               ROOT / 'scripts/type2branch_reference_smoke.py']
    plan = {'scope': 'Frozen encoder inference only on paired BEACON TRAIN windows',
            'protocol_sha256': PROTOCOL_SHA256, 'checkpoint': chosen, 'batch_size': 32,
            'cpu_threads': 2, 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
            'input_sha256': {str((PREP / 'features.npz').relative_to(ROOT)): prepared['features_sha256']}}
    # Explicitly pin the author model, configuration, and compatibility backend.
    plan['source_sha256'].update(original['source_sha256'])
    plan['source_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in pieces})
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    identities = [subject for group in protocol['identity_groups'] for subject in group]
    with np.load(PREP / 'features.npz', allow_pickle=False) as arrays:
        x, metadata = validate_arrays(arrays, identities)
    os.environ.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='', TF_NUM_INTEROP_THREADS='2',
                      TF_NUM_INTRAOP_THREADS='2', TF_CPP_MIN_LOG_LEVEL='2',
                      KERAS_HOME=str(ROOT / '.research-tmp/type2branch-keras'),
                      XDG_CACHE_HOME=str(ROOT / '.research-tmp/type2branch-cache'))
    sys.path.insert(0, str(ROOT / '.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__ != '1.26.4':
        raise ValueError('Pinned NumPy1.26.4 required')
    ref = ROOT / 'research/benchmarks/references/type2branch'
    module('conf', ref / 'conf.small.1Kusers.py')
    model = module('beacon_type2branch_model', ref / 'model.py').get_model_Type2Branch(
        {'SEQUENCE_LENGTH': 100, 'INPUT_FEATURES': 5})['model']
    reader = tf.train.load_checkpoint(str(checkpoint))
    key = 'optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
    if not reader.has_tensor(key) or int(reader.get_tensor(key)) != 500:
        raise ValueError('Saved optimizer iteration mismatch')
    restored = tf.train.Checkpoint(model=model).restore(str(checkpoint))
    restored.assert_existing_objects_matched()
    restored.expect_partial()
    started = time.perf_counter()
    embeddings = np.concatenate([model(x[i:i+32], training=False).numpy() for i in range(0, len(x), 32)])
    elapsed = time.perf_counter() - started
    if embeddings.shape != (len(x), 256) or not np.isfinite(embeddings).all():
        raise ValueError('Finite Nx256 embeddings required')
    verify_hashes(plan)
    verify_hashes(protocol)
    verify_hashes(prep_plan)
    target = OUT / 'embeddings.npz'
    np.savez_compressed(target, embeddings=embeddings, **metadata)
    report = {'status': 'paired_train_embeddings_complete', 'embeddings_sha256': sha(target),
              'features_sha256': prepared['features_sha256'], 'rows': len(x),
              'embedding_shape': list(embeddings.shape), 'seconds': elapsed,
              'selected_checkpoint': chosen, 'below25_windows': int((metadata['true_length'] < 25).sum()),
              'dev_test_payloads_decoded': False, 'model_updates': 0}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
