"""Generated-coordinate checks; no dataset or participant recordings accessed."""
import numpy as np
import pytest
from engine.pointer_sequence import displacement_blocks, enroll_sequence_profile


def test_pointer_window_boundaries_and_coordinate_invariance():
    t = np.arange(257)
    points = np.column_stack([t ** 1.2, np.sin(t / 5)])
    original = points.copy()
    blocks = displacement_blocks(points)
    assert blocks.shape == (2, 2, 128)
    assert displacement_blocks(points[:128]).shape == (0, 2, 128)
    assert displacement_blocks(points[:129]).shape == (1, 2, 128)
    np.testing.assert_allclose(blocks, displacement_blocks(points * 5 + 900), atol=1e-6)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(blocks.mean(axis=(1, 2)), 0, atol=1e-6)
    np.testing.assert_allclose(blocks.std(axis=(1, 2)), 1, atol=1e-6)


def test_pointer_invalid_coordinates_rejected():
    with pytest.raises(ValueError, match='finite'):
        displacement_blocks([[1, np.nan]])


def test_pointer_account_enrollment_and_incomplete_probe():
    pytest.importorskip('sklearn')
    rng = np.random.default_rng(12)
    enrollment = rng.normal(size=(20, 128))
    profile = enroll_sequence_profile(enrollment, nu=.1)
    assert profile.enrollment_blocks == 20
    expected = profile.model.score_samples(enrollment) / 2
    np.testing.assert_allclose(profile.score_blocks(enrollment), expected)
    assert profile.verify(enrollment[:2], threshold=.1, blocks_per_decision=3) == []
    assert len(profile.verify(enrollment[:6], threshold=.1, blocks_per_decision=3)) == 2
    with pytest.raises(ValueError, match='threshold'):
        profile.verify(enrollment, threshold=np.nan, blocks_per_decision=1)
    with pytest.raises(ValueError, match='At least two'):
        enroll_sequence_profile(enrollment[:1])


def test_latent_profiles_support_unseen_users_and_score_direction():
    enrollment = np.ones((3, 128))
    genuine = np.ones((1, 128))
    impostor = -np.ones((1, 128))
    for method in ['cosine', 'latent_manhattan']:
        profile = enroll_sequence_profile(enrollment, method=method)
        assert profile.score_blocks(genuine)[0] > profile.score_blocks(impostor)[0]
        assert profile.verify(genuine, threshold=-.1, blocks_per_decision=1)[0]['matched']
        assert not profile.verify(impostor, threshold=-.1, blocks_per_decision=1)[0]['matched']
