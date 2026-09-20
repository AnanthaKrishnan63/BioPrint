"""Generated feature archives/checkpoints only; never read participant data."""
import hashlib
import json
import sys
import numpy as np
import pytest
import torch
import hmog_select as select


def arrays():
    return {'key': np.zeros((4, 50, 10), np.float32),
            'imu': np.zeros((4, 100, 24), np.float32),
            'subject': np.array(['180679', '622852', '180679', '622852']),
            'session': np.array([1, 1, 9, 9], np.int16),
            'role': np.array(['train_enrollment'] * 2 + ['train_selection'] * 2)}


def save(tmp_path, values):
    path = tmp_path / 'synthetic-features.npz'
    np.savez(path, **values)
    return path


def coverage(values):
    result = {}
    for sid in sorted(select.SELECTION_IDS):
        result[sid] = {}
        for session in range(1, 17):
            n = int(np.sum((values['subject'] == sid) & (values['session'] == session)))
            result[sid][str(session)] = {'candidate_windows': n + 1, 'eligible_windows': n}
    return {'subjects': result}


def test_load_valid_and_metadata_before_feature_values(tmp_path):
    values = arrays()
    loaded = select.load_selection_arrays(save(tmp_path, values))
    for a, name in zip(loaded, ['key', 'imu', 'subject', 'session', 'role']):
        np.testing.assert_array_equal(a, values[name])
    values['subject'][0] = '556357'
    values['key'] = np.zeros(1, dtype=object)  # Must reject identity before opening this.
    with pytest.raises(PermissionError, match='Non-selection'):
        select.load_selection_arrays(save(tmp_path, values))


@pytest.mark.parametrize('field,value', [
    ('subject', np.array(['180679', '622852', '717868', '622852'])),
    ('role', np.array(['train_enrollment'] * 2 + ['dev_probe', 'train_selection'])),
    ('session', np.array([1, 1, 17, 9])),
])
def test_rejects_unauthorized_role_identity_or_session(tmp_path, field, value):
    values = arrays(); values[field] = value
    with pytest.raises(PermissionError):
        select.load_selection_arrays(save(tmp_path, values))


@pytest.mark.parametrize('field,value', [
    ('subject', np.array([180679, 622852, 180679, 622852])),
    ('session', np.array([1., 1., 9., 9.])),
    ('role', np.array([['train_enrollment']] * 4)),
    ('key', np.zeros((4, 50, 10), np.float64)),
    ('imu', np.zeros((4, 99, 24), np.float32)),
    ('key', np.full((4, 50, 10), np.nan, np.float32)),
])
def test_rejects_invalid_feature_contract(tmp_path, field, value):
    values = arrays(); values[field] = value
    with pytest.raises(ValueError):
        select.load_selection_arrays(save(tmp_path, values))


def test_missing_account_and_coverage_cannot_silently_drop():
    values = arrays()
    assert not select.cohort_feasibility(values['subject'], values['role'])
    missing = select.cohort_feasibility(values['subject'][:3], values['role'][:3])
    assert missing == [{'subject': '622852', 'missing_role': 'train_selection'}]
    report = coverage(values)
    result = select.coverage_from_report(report, values['subject'], values['session'])
    assert result['180679']['train_selection']['missing_windows'] == 8
    report['subjects']['180679']['9']['eligible_windows'] = 0
    with pytest.raises(ValueError, match='counts'):
        select.coverage_from_report(report, values['subject'], values['session'])


def test_scoring_all_claims_and_joint_only_earliest_tie():
    values = arrays()
    emb = np.zeros((4, 64)); emb[:, 0] = [0, 10, 1, 9]
    metrics, scores = select.score_embeddings({b: emb for b in select.BRANCHES},
                                              values['subject'], values['role'])
    assert metrics['joint'] == {'pooled_eer': 0., 'genuine_count': 2, 'impostor_count': 2}
    np.testing.assert_array_equal(scores['joint_distances'], [[1, 9], [9, 1]])
    history = [{'epoch': 3, 'metrics': {'joint': {'pooled_eer': .2}, 'key': {'pooled_eer': 0}}},
               {'epoch': 1, 'metrics': {'joint': {'pooled_eer': .2}, 'key': {'pooled_eer': 1}}}]
    assert select.select_epoch(history) == 1


def test_metadata_empty_session_keeps_unknown_window_count():
    values = arrays(); report = coverage(values)
    del report['subjects']['180679']['2']
    report['excluded_sessions'] = {'180679': {'2': {
        'reason': 'empty_keypress', 'basis': 'archive_metadata',
        'activity_contents_inspected': False, 'candidate_windows': None,
        'eligible_windows': None}}}
    result = select.coverage_from_report(report, values['subject'], values['session'])
    enrollment = result['180679']['train_enrollment']
    assert enrollment['candidate_windows'] == 8
    assert enrollment['excluded_session_candidate_windows'] is None
    assert 2 not in enrollment['inspected_sessions']
    assert enrollment['excluded_sessions']['2']['reason'] == 'empty_keypress'
    report['excluded_sessions']['180679']['2']['candidate_windows'] = 0
    with pytest.raises(ValueError, match='metadata-only'):
        select.coverage_from_report(report, values['subject'], values['session'])


def test_unexplained_missing_session_and_excluded_actual_feature_rejected():
    values = arrays(); report = coverage(values)
    del report['subjects']['180679']['1']
    with pytest.raises(ValueError, match='all sessions'):
        select.coverage_from_report(report, values['subject'], values['session'])
    report['excluded_sessions'] = {'180679': {'1': {
        'reason': 'missing_keypress', 'basis': 'archive_metadata'}}}
    with pytest.raises(ValueError, match='metadata-only'):
        select.coverage_from_report(report, values['subject'], values['session'])


def test_hash_is_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pt'; path.write_bytes(b'not a checkpoint')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('deserializer called'))
    with pytest.raises(ValueError, match='checksum'):
        select.load_checkpoint(torch.nn.Linear(1, 1), path, '0' * 64)


def test_finite_generated_state_loaded_strictly_cpu(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / 'synthetic.pt'; torch.save(model.state_dict(), path)
    clone = torch.nn.Linear(1, 1)
    select.load_checkpoint(clone, path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert not clone.training
    assert torch.equal(clone.weight, model.weight)
    torch.save({'weight': torch.full((1, 1), float('nan')), 'bias': torch.zeros(1)}, path)
    with pytest.raises(ValueError, match='finite'):
        select.load_checkpoint(clone, path, hashlib.sha256(path.read_bytes()).hexdigest())


def training_report(protocol_sha):
    return {'plan': {'model_protocol_sha256': protocol_sha,
                     'held_out_measurements_accessed': False,
                     'epochs': 20, 'batches_per_epoch': 40, 'triplets_per_batch': 8},
            'validation_performed': False, 'all_parameters_finite': True,
            'checkpoints': [{'epoch': n, 'checkpoint': f'epoch_{n:02d}.pt', 'sha256': 'a' * 64}
                            for n in range(1, 21)]}


@pytest.mark.parametrize('change', ['missing', 'path', 'order', 'protocol', 'heldout'])
def test_frozen_training_candidate_contract(change):
    report = training_report('hash')
    if change == 'missing': report['checkpoints'].pop()
    elif change == 'path': report['checkpoints'][0]['checkpoint'] = '../epoch_01.pt'
    elif change == 'order': report['checkpoints'].reverse()
    elif change == 'protocol': report['plan']['model_protocol_sha256'] = 'wrong'
    else: report['plan']['held_out_measurements_accessed'] = True
    with pytest.raises(ValueError): select.validate_training_report(report, 'hash')


@pytest.mark.parametrize('infeasible', [False, True])
def test_full_generated_selection_pins_before_values_copies_exact_winner(tmp_path, monkeypatch, infeasible):
    monkeypatch.setattr(select, 'ROOT', tmp_path)
    protocol = tmp_path / 'protocol.json'; protocol.write_text('{"version": 2}')
    monkeypatch.setattr(select, 'PROTOCOL', protocol)
    source_dir = tmp_path / 'scripts'; source_dir.mkdir()
    for name in ['hmog_select.py', 'hmog_verification.py', 'hmog_training_core.py']:
        (source_dir / name).write_text('# generated provenance fixture\n')
    training_dir = tmp_path / 'training'; training_dir.mkdir()
    report = training_report(hashlib.sha256(protocol.read_bytes()).hexdigest())
    class TinyModel(torch.nn.Module):
        def __init__(self, *args):
            super().__init__(); self.value = torch.nn.Parameter(torch.zeros(1))
    for row in report['checkpoints']:
        path = training_dir / row['checkpoint']
        torch.save({'value': torch.tensor([float(row['epoch'])])}, path)
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (training_dir / 'training_complete.json').write_text(json.dumps(report))
    values = arrays()
    if infeasible:
        values = {k: v[:3] for k, v in values.items()}
    feature_path = save(tmp_path, values)
    extraction = coverage(values)
    extraction.update(feature_archive_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                      held_out_measurements_read=False)
    extraction_path = tmp_path / 'extraction.json'; extraction_path.write_text(json.dumps(extraction))
    output = tmp_path / 'selection'
    loader = select.load_selection_arrays
    def checked_loader(path):
        assert (output / 'selection_plan.json').exists()
        assert (output / 'source_hmog_select.py').exists()
        return loader(path)
    monkeypatch.setattr(select, 'load_selection_arrays', checked_loader)
    monkeypatch.setattr(select, 'load_author_classes', lambda: (TinyModel, None))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    def fake_embeddings(model, key, imu):
        emb = np.zeros((4, 64)); emb[:, 0] = [0, 10, 1, 9]
        return {b: emb.copy() for b in select.BRANCHES}
    monkeypatch.setattr(select, 'embed_all', fake_embeddings)
    monkeypatch.setattr(sys, 'argv', ['hmog_select', '--training-dir', str(training_dir),
                        '--features', str(feature_path), '--extraction-report', str(extraction_path),
                        '--output', str(output)])
    select.main()
    completed = json.loads((output / 'selection_complete.json').read_text())
    if infeasible:
        assert completed['status'] == 'infeasible'
        assert completed['missing'] == [{'subject': '622852', 'missing_role': 'train_selection'}]
        assert not (output / 'selected_checkpoint.pt').exists()
        assert not (output / 'score_history.jsonl').exists()
        return
    assert completed['selected_epoch'] == 1
    assert len(completed['history']) == 20
    assert (output / 'selected_checkpoint.pt').read_bytes() == (training_dir / 'epoch_01.pt').read_bytes()
    with np.load(output / 'selected_scores.npz', allow_pickle=False) as scores:
        assert set(scores.files) == {'probe_subject', 'accounts', 'genuine_mask',
                                    'joint_distances', 'key_distances', 'imu_distances'}
    with pytest.raises(FileExistsError): select.main()
