"""Generated selection metadata/tensors/scores only; no participant data reads."""
import json
from argparse import Namespace
import numpy as np
import pytest
import hmog_tap_selection as runner


def generated():
    subjects = np.array(['180679'] * 8 + ['622852'] * 52)
    roles = np.array(['train_enrollment'] * 5 + ['train_selection'] * 3 +
                     ['train_enrollment'] * 10 + ['train_selection'] * 42)
    sessions = np.where(roles == 'train_enrollment', 1, 9).astype(np.int16)
    return {'subject': subjects, 'session': sessions, 'role': roles,
            'activity_id': np.arange(60, dtype=np.int64), 'window_index': np.zeros(60, np.int16)}


def archives(tmp_path, meta):
    base = dict(meta, key=np.zeros((60, 50, 10), np.float32), imu=np.zeros((60, 100, 24), np.float32))
    taps = np.repeat((np.arange(1200) % 2)[:, None] * 2., 11, axis=1)
    tap = {'window_' + k: v.copy() for k, v in meta.items()}
    tap.update(tap_vectors=taps, tap_window=np.repeat(np.arange(60), 20),
               window_tap_count=np.full(60, 20), window_scorable=np.ones(60, bool))
    a, b = tmp_path / 'synthetic-base.npz', tmp_path / 'synthetic-tap.npz'
    np.savez(a, **base); np.savez(b, **tap)
    return a, b, base, tap


def test_load_exact_generated_window_alignment_and_personal_enrollment(tmp_path):
    paths = archives(tmp_path, generated())
    meta, _, _, taps, index = runner.load_paired(*paths[:2])
    gallery, profiles, diagnostics, missing = runner.personal_galleries(meta, taps, index)
    assert gallery.sum() == 15 and not missing
    assert diagnostics['180679']['enrollment_taps'] == 100
    assert diagnostics['622852']['enrollment_taps'] == 200
    assert profiles['180679']['mean'] == [1.] * 11


@pytest.mark.parametrize('case', ['foreign', 'test', 'role', 'duplicate', 'counts', 'float_session'])
def test_metadata_guards_before_paired_feature_values(case):
    meta = generated()
    if case == 'foreign': meta['subject'][0] = '717868'
    elif case == 'test': meta['session'][0] = 17
    elif case == 'role': meta['role'][0] = 'dev_probe'
    elif case == 'duplicate': meta['activity_id'][1] = meta['activity_id'][0]
    elif case == 'counts': meta['subject'][0] = '622852'
    else: meta['session'] = meta['session'].astype(float)
    with pytest.raises((ValueError, PermissionError)): runner.validate_metadata(meta)


def test_misaligned_metadata_rejected_before_poisoned_feature_access(tmp_path):
    a, b, base, tap = archives(tmp_path, generated())
    tap['window_activity_id'][0] += 100
    base['key'] = np.array([object()], dtype=object)
    np.savez(a, **base); np.savez(b, **tap)
    with pytest.raises(ValueError, match='metadata mismatch'): runner.load_paired(a, b)


def test_missing_tap_gallery_returns_no_profile_without_borrowing_probes(tmp_path):
    a, b, _, _ = archives(tmp_path, generated())
    meta, _, _, taps, index = runner.load_paired(a, b)
    keep = ~((meta['subject'][index] == '180679') & (meta['role'][index] == 'train_enrollment'))
    gallery, profiles, _, missing = runner.personal_galleries(meta, taps[keep], index[keep])
    assert missing == ['180679'] and set(profiles) == {'622852'}
    assert not gallery[:8].any()


def test_fixed_scales_applied_without_any_refitting_and_same_gallery(tmp_path):
    a, b, _, _ = archives(tmp_path, generated())
    meta, _, _, taps, index = runner.load_paired(a, b)
    gallery, profiles, _, _ = runner.personal_galleries(meta, taps, index)
    embeddings = {}
    for branch in runner.BRANCHES:
        values = np.zeros((60, 64)); values[:, 0] = np.where(meta['subject'] == '180679', 0., 10.)
        embeddings[branch] = values
    frozen = {'scales': {'joint_tap11': [2., 4.], 'key_imu_tap11': [2., 5., 4.]}}
    scores, masks = runner.paired_scores(embeddings, meta, taps, index, gallery, profiles, frozen)
    assert all(x.shape == (60, 2) for x in scores.values())
    assert all(m.all() for m in masks.values())
    assert scores['joint_tap11'][0, 1] == 2.5
    assert scores['key_imu_tap11'][0, 1] == pytest.approx(7 / 3)
    assert frozen['scales']['joint_tap11'] == [2., 4.]


def test_ranking_uses_shared_mask_but_retains_all_original_probe_denominators():
    meta = generated(); genuine = meta['subject'][:, None] == np.array(runner.ACCOUNTS)
    scores = {m: np.where(genuine, 0., 5.) for m in runner.METHODS}
    masks = {m: np.ones((60, 2), bool) for m in runner.METHODS}
    masks['tap11'][5:8] = False
    thresholds = {m: {'0.001': 1., '0.01': 1., '0.05': 1.} for m in runner.METHODS}
    result = runner.summarize(scores, masks, meta['subject'], runner.ACCOUNTS,
                              meta['role'] == 'train_selection', thresholds)
    assert result['selected_method'] == 'joint_tap11'
    assert result['macro_genuine_not_accepted'] == .5
    shared = result['metrics']['joint_tap11']['shared_intersection']
    assert shared['per_account_at_1pct']['180679']['expected_genuine'] == 3
    assert shared['per_account_at_1pct']['622852']['expected_genuine'] == 42
    assert result['metrics']['joint_tap11']['base_coverage']['macro_genuine_not_accepted'] == 0.
    assert result['probe_windows'] == 45


def test_changed_canonical_paths_rejected_before_data_loading(tmp_path, monkeypatch):
    args = Namespace(**{k: tmp_path / k for k in runner.PATH_ARGS})
    args.output.mkdir()
    (args.output / 'selection_plan.json').write_text(json.dumps({'canonical_paths': {}}))
    monkeypatch.setattr(runner, 'load_paired', lambda *a: pytest.fail('Unbound data read'))
    with pytest.raises(ValueError, match='CLI paths'): runner.run(args)


@pytest.mark.parametrize('wrong_protocol', [False, True])
def test_prepare_binds_extraction_protocol_without_decoding_arrays(tmp_path, monkeypatch, wrong_protocol):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    args = Namespace(**{k: tmp_path / k for k in runner.PATH_ARGS})
    args.paired_dir.mkdir(); args.selection_dir.mkdir(); (tmp_path / 'scripts').mkdir()
    for name in runner.SOURCES: (tmp_path / 'scripts' / name).write_text('# generated source\n')
    args.features.write_bytes(b'not an NPZ; prepare must not decode')
    args.tap_features.write_bytes(b'also not an NPZ')
    checkpoint = args.selection_dir / 'selected_checkpoint.pt'; checkpoint.write_bytes(b'generated model')
    model_sha = runner.digest(checkpoint)
    frozen = {'status': 'complete', 'plan': {'checkpoint_sha256': model_sha},
              'scales': {'joint_tap11': [1., 1.], 'key_imu_tap11': [1., 1., 1.]},
              'thresholds': {m: {'0.001': 1., '0.01': 2., '0.05': 3.} for m in runner.METHODS}}
    paired_path = args.paired_dir / 'paired_train_complete.json'; paired_path.write_text(json.dumps(frozen))
    paired_key = 'research/benchmarks/hmog/tap_paired_train_v2/paired_train_complete.json'
    protocol_paired = tmp_path / paired_key; protocol_paired.parent.mkdir(parents=True)
    protocol_paired.write_bytes(paired_path.read_bytes())
    protocol = {'methods': list(runner.METHODS), 'identities': runner.ACCOUNTS,
                'operating_target': .01, 'status': 'Frozen root-approved generated fixture',
                'source_hashes': {paired_key: runner.digest(paired_path)},
                'selection_features': {'sha256': runner.digest(args.features)}, 'caveats': ['generated']}
    args.protocol.write_text(json.dumps(protocol))
    args.tap_report.write_text(json.dumps({'feature_archive_sha256': runner.digest(args.tap_features),
        'held_out_measurements_read': False, 'plan': {
            'proposal_sha256': 'wrong' if wrong_protocol else runner.digest(args.protocol),
            'selection_window_archive_sha256': runner.digest(args.features)}}))
    (args.selection_dir / 'selection_complete.json').write_text(json.dumps({
        'status': 'complete', 'selected_epoch': 3, 'selected_checkpoint_sha256': model_sha}))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Prepare decoded feature values'))
    if wrong_protocol:
        with pytest.raises(ValueError, match='extraction'): runner.prepare(args)
        assert not args.output.exists()
    else:
        runner.prepare(args)
        plan = json.loads((args.output / 'selection_plan.json').read_text())
        assert plan['canonical_paths'] == runner.canonical_paths(args)
        assert plan['thresholds'] == frozen['thresholds'] and plan['scales'] == frozen['scales']
        assert (args.output / 'source_hmog_tap_selection.py').exists()
        with pytest.raises(FileExistsError): runner.prepare(args)
