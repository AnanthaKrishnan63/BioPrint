"""Generated DEV-shaped fixtures only; no actual participant arrays or inference."""
from argparse import Namespace
import json
import numpy as np
import pytest
import hmog_tap_dev as runner


def generated():
    subjects, roles = [], []
    for sid, (support, probe) in runner.COUNTS.items():
        subjects += [sid] * (support + probe)
        roles += ['train_support'] * support + ['dev_probe'] * probe
    return {'subject': np.array(subjects), 'role': np.array(roles),
            'session': np.where(np.array(roles) == 'train_support', 1, 9).astype(np.int16),
            'activity_id': np.arange(68, dtype=np.int64), 'window_index': np.zeros(68, np.int16)}


def archives(tmp_path):
    meta = generated()
    base = dict(meta, key=np.zeros((68, 50, 10), np.float32), imu=np.zeros((68, 100, 24), np.float32))
    tap = {'window_' + k: v.copy() for k, v in meta.items()}
    tap.update(tap_vectors=np.repeat((np.arange(6800) % 2)[:, None] * 2., 11, axis=1),
               tap_window=np.repeat(np.arange(68), 100), window_tap_count=np.full(68, 100),
               window_scorable=np.ones(68, bool))
    a, b = tmp_path / 'synthetic-base.npz', tmp_path / 'synthetic-tap.npz'
    np.savez(a, **base); np.savez(b, **tap)
    return a, b, base, tap


def test_exact68_metadata_and_support_only_allfour_personal_galleries(tmp_path):
    a, b, _, _ = archives(tmp_path)
    meta, _, _, taps, index = runner.load_paired(a, b)
    gallery, profiles, diagnostics, missing = runner.personal_galleries(meta, taps, index)
    assert gallery.sum() == 37 and not missing and set(profiles) == set(runner.ACCOUNTS)
    assert diagnostics['556357']['enrollment_taps'] == 100
    assert diagnostics['737973']['enrollment_taps'] == 100
    assert runner.coverage(meta, index, gallery)['accounts']['556357']['dev_probe']['original_windows'] == 0


@pytest.mark.parametrize('case', ['test', 'fit_identity', 'probe_as_support', 'duplicate', 'float_session'])
def test_strict_devmetadata_guards(case):
    meta = generated()
    if case == 'test': meta['session'][0] = 17
    elif case == 'fit_identity': meta['subject'][0] = '717868'
    elif case == 'probe_as_support': meta['role'][24] = 'train_support'
    elif case == 'duplicate': meta['activity_id'][1] = meta['activity_id'][0]
    else: meta['session'] = meta['session'].astype(float)
    with pytest.raises((ValueError, PermissionError)): runner.validate_metadata(meta)


def test_poisoned_features_not_opened_on_pairing_mismatch(tmp_path):
    a, b, base, tap = archives(tmp_path)
    tap['window_window_index'][0] += 1; base['key'] = np.array([object()], dtype=object)
    np.savez(a, **base); np.savez(b, **tap)
    with pytest.raises(ValueError, match='metadata mismatch'): runner.load_paired(a, b)


def test_sub80_singlewindow_gallery_stays_missing_not_repaired(tmp_path):
    a, b, _, _ = archives(tmp_path)
    meta, _, _, taps, index = runner.load_paired(a, b)
    target = np.flatnonzero(meta['subject'] == '556357')[0]
    keep = np.ones(len(taps), bool); keep[np.flatnonzero(index == target)[79:]] = False
    gallery, profiles, diagnostics, missing = runner.personal_galleries(meta, taps[keep], index[keep])
    assert gallery.sum() == 37
    assert missing == ['556357'] and '556357' not in profiles
    assert diagnostics['556357']['enrollment_taps'] == 79


def test_allfour_claims_fixedscales_and_zero_probe_accounts_retained(tmp_path):
    from hmog_tap_dev_metrics import summarize
    a, b, _, _ = archives(tmp_path)
    meta, _, _, taps, index = runner.load_paired(a, b)
    gallery, profiles, _, _ = runner.personal_galleries(meta, taps, index)
    values = np.zeros((68, 64))
    for n, sid in enumerate(runner.ACCOUNTS): values[meta['subject'] == sid, 0] = n * 10
    embeddings = {b: values.copy() for b in runner.BRANCHES}
    frozen = {'scales': {'joint_tap11': [2., 4.], 'key_imu_tap11': [2., 5., 4.]}}
    scores, available = runner.paired_scores(embeddings, meta, taps, index, gallery, profiles, frozen)
    assert all(s.shape == (68, 4) for s in scores.values())
    assert scores['joint_tap11'][0, 1] == 2.5
    thresholds = {m: {'0.001': 1., '0.01': 1., '0.05': 1.} for m in runner.METHODS}
    result = summarize(scores, available, meta['subject'], runner.ACCOUNTS,
                       meta['role'] == 'dev_probe', thresholds, expected_candidate_probes=154)
    assert result['status'] == 'infeasible' and result['observed_window_comparison_completed']
    assert result['missing_genuine_probe_accounts'] == ['556357', '737973']
    assert 'selected_method' not in result
    metric = result['metrics']['joint']['base_coverage']
    assert metric['base_window_rates']['0.01']['impostor_count'] == 93
    assert metric['known_inspected_candidate_rates']['0.01']['missing_genuine'] == 123
    assert metric['per_account']['556357']['rates']['0.01']['frr'] is None


def test_known_candidate_coverage_must_be_exact_not_inferred_zero():
    report = {'coverage': {a: {'dev_probe': {'candidate_windows': n}}
                          for a, n in zip(runner.ACCOUNTS, [50, 30, 30, 44])}}
    assert runner.upstream_candidate_count(report) == 154
    report['coverage']['556357']['dev_probe']['candidate_windows'] = None
    with pytest.raises(ValueError): runner.upstream_candidate_count(report)


def test_changed_paths_and_hashes_reject_before_dev_values(tmp_path, monkeypatch):
    args = Namespace(**{k: tmp_path / k for k in runner.PATH_ARGS}); args.output.mkdir()
    (args.output / 'dev_plan.json').write_text(json.dumps({'canonical_paths': {}}))
    monkeypatch.setattr(runner, 'load_paired', lambda *a: pytest.fail('Unbound DEV values read'))
    with pytest.raises(ValueError, match='CLI paths'): runner.run(args)


def test_missing_gallery_stops_before_encoder_and_records_no_metrics(tmp_path, monkeypatch):
    a, b, _, _ = archives(tmp_path)
    meta, key, imu, taps, index = runner.load_paired(a, b)
    target = np.flatnonzero(meta['subject'] == '556357')[0]
    keep = np.ones(len(taps), bool)
    keep[np.flatnonzero(index == target)[77:]] = False
    args = Namespace(**{k: tmp_path / k for k in runner.PATH_ARGS})
    args.output.mkdir()
    plan = {'canonical_paths': runner.canonical_paths(args), 'input_sha256': {}, 'source_sha256': {}}
    (args.output / 'dev_plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'load_paired', lambda *args: (meta, key, imu, taps[keep], index[keep]))
    monkeypatch.setattr(runner, 'load_author_classes', lambda: pytest.fail('Encoder must not run'))
    runner.run(args)
    result = json.loads((args.output / 'dev_complete.json').read_text())
    assert result['status'] == 'infeasible'
    assert result['missing_gallery_accounts'] == ['556357']
    assert result['profile_diagnostics']['556357']['enrollment_taps'] == 77
    assert result['metrics'] == {}
    assert not (args.output / 'dev_scores.npz').exists()
