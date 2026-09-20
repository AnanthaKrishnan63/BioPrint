import numpy as np
import pytest

from hmog_score_fusion import fit_scales, fuse


def test_training_mask_excludes_genuine_and_heldout_claims():
    scores = np.array([[[2., 4.], [6., 12.]], [[900., 800.], [700., 600.]]])
    mask = np.ones_like(scores, dtype=bool)
    train = np.array([[True, True], [False, False]])
    scales = fit_scales(scores, mask, train)
    np.testing.assert_array_equal(scales, [4., 8.])
    scores[1] *= 100
    np.testing.assert_array_equal(fit_scales(scores, mask, train), scales)
    result, complete = fuse(scores[:1], mask[:1], scales)
    np.testing.assert_array_equal(result, [[.5, 1.5]])
    assert complete.all()


def test_missing_modality_never_becomes_partial_fusion():
    scores = np.array([[[2., 4.], [6., np.nan]]])
    available = np.isfinite(scores)
    scales = fit_scales(scores, available, np.array([[True, True]]))
    np.testing.assert_array_equal(scales, [2., 4.])
    result, complete = fuse(scores, available, scales)
    assert result[0, 0] == 1
    assert np.isnan(result[0, 1])
    assert complete.tolist() == [[True, False]]


def test_empty_training_claims_fail():
    with pytest.raises(ValueError, match='No complete'):
        fit_scales(np.ones((1, 1, 2)), np.ones((1, 1, 2), dtype=bool),
                   np.zeros((1, 1), dtype=bool))


def test_zero_scale_fails_without_modality_drop():
    with pytest.raises(ValueError, match='positive TRAIN'):
        fit_scales(np.array([[[0., 1.]]]), np.ones((1, 1, 2), dtype=bool),
                   np.ones((1, 1), dtype=bool))


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_invalid_observed_distance_rejected(value):
    with pytest.raises(ValueError):
        fuse(np.array([[[value, 1.]]]), np.ones((1, 1, 2), dtype=bool), [1., 1.])
