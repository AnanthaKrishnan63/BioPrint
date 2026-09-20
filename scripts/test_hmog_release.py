"""Generated release fixtures only; no actual participant features or checkpoints."""
import hashlib
import json
import asyncio
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import hmog_release as release


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def generated(tmp_path, monkeypatch):
    root = tmp_path
    base = root / 'research/benchmarks/hmog'
    validation = base / 'validation_v1'
    selected = base / 'selection_v1'
    calibration = base / 'calibration_fallback_v1'
    scripts = root / 'scripts'
    for directory in (validation, selected, calibration, scripts):
        directory.mkdir(parents=True)
    for name in ('hmog_training_core.py', 'hmog_verification.py'):
        (scripts / name).write_text('# generated source fixture\n')
    monkeypatch.setattr(release, 'ROOT', root)
    monkeypatch.setattr(release, 'BASE', base)
    monkeypatch.setattr(release, 'VALIDATION', validation)
    monkeypatch.setattr(release, 'RELEASE', base / 'release_v1')
    accounts = np.array(sorted(('556357', '219303', '777078', '737973')))
    features = {'key': np.zeros((4, 50, 10), np.float32),
                'imu': np.zeros((4, 100, 24), np.float32),
                'subject': accounts.copy(), 'session': np.array([9, 10, 15, 16], np.int16),
                'role': np.array(['dev_probe'] * 4)}
    profiles = {'accounts': accounts.copy(), **{b: np.zeros((4, 64)) for b in release.BRANCHES}}
    np.savez(validation / 'dev_features.npz', **features)
    np.savez(validation / 'profiles.npz', **profiles)
    (selected / 'selected_checkpoint.pt').write_bytes(b'generated checkpoint')
    (calibration / 'thresholds.json').write_text(json.dumps({
        b: {'0.001': .1, '0.01': .2, '0.05': .3} for b in release.BRANCHES}))
    report = {'status': 'complete', 'missing': [], 'test_measurements_read': False,
              'validation_only': True, 'profiles_sha256': sha(validation / 'profiles.npz'),
              'dev_features_sha256': sha(validation / 'dev_features.npz'),
              'plan': {'selected_checkpoint_sha256': sha(selected / 'selected_checkpoint.pt'),
                       'thresholds_sha256': sha(calibration / 'thresholds.json')}}
    def persist():
        (validation / 'validation_complete.json').write_text(json.dumps(report))
    persist()
    return dict(root=root, base=base, validation=validation, selected=selected,
                calibration=calibration, report=report, persist=persist,
                features=features, profiles=profiles)


@pytest.mark.parametrize('field,value', [('status', 'infeasible'),
    ('missing', [{'subject': '556357', 'missing_role': 'train_support'}]),
    ('test_measurements_read', True), ('validation_only', False)])
def test_infeasible_or_unauthorized_validation_never_exports(generated, field, value):
    generated['report'][field] = value; generated['persist']()
    with pytest.raises(ValueError, match='Complete four-account'):
        release.export(generated['calibration'])
    assert not release.RELEASE.exists()


@pytest.mark.parametrize('case', ['support_role', 'support_session', 'test_session', 'foreign_id'])
def test_hash_bound_forbidden_probe_metadata_never_exports(generated, case):
    data = generated['features']
    if case == 'support_role':
        data['role'] = data['role'].astype('U32'); data['role'][0] = 'train_support'
    elif case == 'support_session': data['session'][0] = 8
    elif case == 'test_session': data['session'][0] = 17
    else: data['subject'][0] = '717868'
    path = generated['validation'] / 'dev_features.npz'; np.savez(path, **data)
    generated['report']['dev_features_sha256'] = sha(path); generated['persist']()
    with pytest.raises(ValueError, match='DEV probe'):
        release.export(generated['calibration'])
    assert not release.RELEASE.exists()


@pytest.mark.parametrize('filename', ['encoder', 'profiles', 'features', 'thresholds'])
def test_changed_frozen_artifact_refused_before_writes(generated, filename):
    paths = {'encoder': generated['selected'] / 'selected_checkpoint.pt',
             'profiles': generated['validation'] / 'profiles.npz',
             'features': generated['validation'] / 'dev_features.npz',
             'thresholds': generated['calibration'] / 'thresholds.json'}
    paths[filename].write_bytes(paths[filename].read_bytes() + b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        release.export(generated['calibration'])
    assert not release.RELEASE.exists()


@pytest.mark.parametrize('case', ['wrong_order', 'missing_account', 'nonfinite'])
def test_profiles_must_cover_all_four_accounts_in_fixed_order(generated, case):
    profiles = generated['profiles']
    if case == 'wrong_order': profiles['accounts'] = profiles['accounts'][::-1]
    elif case == 'missing_account': profiles['joint'] = profiles['joint'][:3]
    else: profiles['imu'][0, 0] = np.nan
    path = generated['validation'] / 'profiles.npz'; np.savez(path, **profiles)
    generated['report']['profiles_sha256'] = sha(path); generated['persist']()
    with pytest.raises(ValueError): release.export(generated['calibration'])
    assert not release.RELEASE.exists()


def test_success_preserves_exact_bytes_and_refuses_overwrite(generated):
    release.export(generated['calibration'])
    manifest_bytes = (release.RELEASE / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['split'] == 'dev' and manifest['version'] == 1
    assert set(manifest['files']) == {'encoder.pt', 'profiles.npz', 'thresholds.json',
                                      'dev_features.npz', 'dev_report.json'}
    for name, digest in manifest['files'].items():
        assert sha(release.RELEASE / name) == digest
    assert (release.RELEASE / 'encoder.pt').read_bytes() == (
        generated['selected'] / 'selected_checkpoint.pt').read_bytes()
    assert (release.RELEASE / 'dev_report.json').read_bytes() == (
        generated['validation'] / 'validation_complete.json').read_bytes()
    before = {p.name: p.read_bytes() for p in release.RELEASE.iterdir()}
    with pytest.raises(FileExistsError): release.export(generated['calibration'])
    assert before == {p.name: p.read_bytes() for p in release.RELEASE.iterdir()}


def test_preexisting_output_is_never_modified(generated):
    release.RELEASE.mkdir()
    sentinel = release.RELEASE / 'keep.txt'; sentinel.write_text('existing work')
    with pytest.raises(FileExistsError): release.export(generated['calibration'])
    assert list(release.RELEASE.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'existing work'


def test_nested_seen_identity_calibration_scope_and_limitation_preserved(generated):
    scope = 'TRAIN seen-identity enrollment calibration; no unseen-identity guarantee'
    limitation = 'Calibration identities also fitted the encoder; operational rates may be optimistic.'
    generated['report']['plan']['calibration_provenance'] = {
        'calibration_scope': scope, 'limitation': limitation,
        'plan_fields': {'calibration_scope': scope},
        'report_fields': {'limitation': limitation}}
    generated['report']['plan']['calibration_scope'] = None
    generated['report']['plan']['calibration_protocol_sha256'] = 'b' * 64
    generated['persist']()
    release.export(generated['calibration'])
    manifest = json.loads((release.RELEASE / 'manifest.json').read_text())
    assert manifest['calibration_scope'] == scope
    assert manifest['calibration_limitation'] == limitation
    assert manifest['calibration_protocol_sha256'] == 'b' * 64


@pytest.mark.parametrize('case', ['float_sessions', 'metadata_shape', 'numeric_subjects',
                                  'key_shape', 'imu_shape', 'key_dtype', 'nonfinite'])
def test_strict_generated_feature_schema_cannot_freeze_unusable_release(generated, case):
    data = generated['features']
    metadata_case = case in ('float_sessions', 'metadata_shape', 'numeric_subjects')
    if case == 'float_sessions': data['session'] = data['session'].astype(float)
    elif case == 'metadata_shape': data['role'] = data['role'].reshape(4, 1)
    elif case == 'numeric_subjects': data['subject'] = data['subject'].astype(int)
    elif case == 'key_shape': data['key'] = data['key'][:, :49]
    elif case == 'imu_shape': data['imu'] = data['imu'][:3]
    elif case == 'key_dtype': data['key'] = data['key'].astype(np.float64)
    else: data['imu'][0, 0, 0] = np.inf
    if metadata_case:
        # If values are read before metadata validation, this raises a different error.
        data['key'] = np.array([object()], dtype=object)
    path = generated['validation'] / 'dev_features.npz'; np.savez(path, **data)
    generated['report']['dev_features_sha256'] = sha(path); generated['persist']()
    message = 'DEV probe' if metadata_case else 'DEV feature contract'
    with pytest.raises(ValueError, match=message): release.export(generated['calibration'])
    assert not release.RELEASE.exists()


def test_replay_report_must_match_frozen_release_before_reading_scores(generated, monkeypatch):
    import research_hmog_api
    release.export(generated['calibration'])
    monkeypatch.setattr(research_hmog_api, 'MANIFEST_SHA256', sha(release.RELEASE / 'manifest.json'))
    # Prevent imports of the actual application; failure must precede any ASGI use.
    monkeypatch.setitem(sys.modules, 'research_api', SimpleNamespace(app=object()))
    generated['report']['replay_sha256'] = '0' * 64
    generated['persist']()
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Changed report reached score loading'))
    with pytest.raises(ValueError, match='differs from frozen API release'):
        asyncio.run(release.replay())
    assert not (generated['base'] / 'api_replay_v1').exists()


def test_replay_requires_explicit_api_manifest_pin(generated, monkeypatch):
    import research_hmog_api
    release.export(generated['calibration'])
    monkeypatch.setattr(research_hmog_api, 'MANIFEST_SHA256', None)
    monkeypatch.setitem(sys.modules, 'research_api', SimpleNamespace(app=object()))
    with pytest.raises(ValueError, match='explicitly pin'):
        asyncio.run(release.replay())
    assert not (generated['base'] / 'api_replay_v1').exists()


def partial_probe_fixture(generated):
    """Two actual generated probe rows, four unchanged support galleries."""
    data = {name: values[[0, 3]] for name, values in generated['features'].items()}
    path = generated['validation'] / 'dev_features.npz'
    np.savez(path, **data)
    present = set(data['subject'])
    report = generated['report']
    report.update(status='infeasible',
                  missing=[{'subject': sid, 'missing_role': 'dev_probe'}
                           for sid in generated['profiles']['accounts'] if sid not in present],
                  dev_features_sha256=sha(path),
                  coverage={sid: {'dev_probe': {'candidate_windows': 5,
                                               'eligible_windows': int(sid in present),
                                               'missing_windows': 5 - int(sid in present)}}
                            for sid in generated['profiles']['accounts']})
    generated['persist']()
    return data


def test_partial_probe_release_default_refused_even_with_all_galleries(generated):
    partial_probe_fixture(generated)
    with pytest.raises(ValueError): release.export(generated['calibration'])
    assert not release.RELEASE.exists()


def test_explicit_partial_export_preserves_missing_coverage_and_all_claim_accounts(generated):
    data = partial_probe_fixture(generated)
    release.export(generated['calibration'], allow_incomplete_probes=True)
    manifest = json.loads((release.RELEASE / 'manifest.json').read_text())
    assert manifest['validation_status'] == 'infeasible'
    assert manifest['missing'] == generated['report']['missing']
    assert manifest['coverage'] == generated['report']['coverage']
    with np.load(release.RELEASE / 'dev_features.npz', allow_pickle=False) as archive:
        for name, values in data.items(): np.testing.assert_array_equal(archive[name], values)
    with np.load(release.RELEASE / 'profiles.npz', allow_pickle=False) as archive:
        assert len(archive['accounts']) == 4
        assert archive['joint'].shape == (4, 64)


@pytest.mark.parametrize('case', ['missing_gallery_role', 'wrong_missing_identity',
                                  'omitted_missing_account', 'duplicate_missing_account',
                                  'empty_missing_map', 'nonfinite_gallery', 'absent_gallery'])
def test_partial_flag_does_not_allow_missing_galleries_or_false_missing_map(generated, case):
    partial_probe_fixture(generated)
    report = generated['report']
    if case == 'missing_gallery_role': report['missing'][0]['missing_role'] = 'train_support'
    elif case == 'wrong_missing_identity': report['missing'][0]['subject'] = '219303'
    elif case == 'omitted_missing_account': report['missing'].pop()
    elif case == 'duplicate_missing_account': report['missing'].append(report['missing'][0].copy())
    elif case == 'empty_missing_map': report['missing'] = []
    else:
        profiles = generated['profiles']
        if case == 'nonfinite_gallery': profiles['joint'][0, 0] = np.nan
        else: profiles['imu'] = profiles['imu'][:3]
        path = generated['validation'] / 'profiles.npz'; np.savez(path, **profiles)
        report['profiles_sha256'] = sha(path)
    generated['persist']()
    with pytest.raises(ValueError):
        release.export(generated['calibration'], allow_incomplete_probes=True)
    assert not release.RELEASE.exists()
