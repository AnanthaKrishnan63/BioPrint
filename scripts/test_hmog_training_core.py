"""Generated metadata/tensors only; never open participant archives."""
import numpy as np
import pytest

from hmog_training_core import TripletSampler, load_author_classes


def test_cross_session_sampling_unequal_availability_and_reproducibility():
    subjects = ['717868'] * 3 + ['526319'] * 4 + ['986737']
    sessions = [9, 9, 12, 10, 11, 14, 14, 13]
    roles = ['train_fit'] * len(subjects)
    first = TripletSampler(subjects, sessions, roles, seed=41).sample(1000)
    second = TripletSampler(subjects, sessions, roles, seed=41).sample(1000)
    np.testing.assert_array_equal(first, second)
    for a, p, n in first:
        assert subjects[a] == subjects[p] != subjects[n]
        assert sessions[a] != sessions[p]
    assert 7 in first[:, 2]  # Single-session account remains a possible negative.
    assert 7 not in first[:, 0]


@pytest.mark.parametrize('subject,session,role', [
    ('717868', 8, 'train_enrollment'), ('717868', 17, 'sealed_test'),
    ('556357', 9, 'dev_probe'), ('180679', 9, 'train_selection'),
    ('962159', 9, 'train_calibration'), ('717868', 9, 'train_support'),
    ('717868', 9.5, 'train_fit'), ('717868', True, 'train_fit'),
])
def test_rejects_non_fit_population_data(subject, session, role):
    with pytest.raises(PermissionError):
        TripletSampler([subject], [session], [role], seed=0)


def test_no_same_session_fallback():
    with pytest.raises(ValueError, match='infeasible'):
        TripletSampler(['717868', '526319'], [9, 10], ['train_fit'] * 2, seed=0)


def test_pinned_exact_loss_formula():
    import torch
    _, loss_class = load_author_classes()
    anchor = torch.tensor([[0., 0.], [0., 0.]])
    positive = torch.tensor([[3., 4.], [0., 1.]])
    negative = torch.tensor([[0., 2.], [0., 4.]])
    assert loss_class(margin=1.)(anchor, positive, negative).item() == 2.


def test_tampered_architecture_denied(tmp_path):
    from hmog_training_core import SOURCE_DIR, SOURCE_PREFIX
    (tmp_path / (SOURCE_PREFIX + 'model.py')).write_text('raise RuntimeError("must not execute")')
    (tmp_path / (SOURCE_PREFIX + 'train.py')).write_bytes(
        (SOURCE_DIR / (SOURCE_PREFIX + 'train.py')).read_bytes())
    with pytest.raises(ValueError, match='architecture checksum'):
        load_author_classes(tmp_path)
