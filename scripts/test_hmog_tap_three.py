"""Generated tap vectors only; no IO, observations, or profile refitting."""
import copy
import numpy as np
import pytest

from hmog_tap_benchmark import CONVENTIONS, build_profiles
from hmog_tap_reference import fit_profile
from hmog_tap_three import FEATURE_INDICES, scaled_manhattan_three, score_windows


def profile(active=range(11)):
    taps = np.ones((80, 11))
    for column in active: taps[:, column] = np.arange(80) % 2 * 2
    return build_profiles(taps, np.zeros(80, int), ['a'], [True], ['a'])[0]['a']


def test_exact_published_indices_sum_and_ignored_remaining_dimensions():
    gallery = profile()
    vector = np.full(11, 1000.); vector[list(FEATURE_INDICES)] = [2., 3., 4.]
    assert FEATURE_INDICES == (0, 1, 10)
    assert scaled_manhattan_three(gallery, vector) == 6.


def test_full11_active_but_selected3_constant_produces_no_verdict():
    gallery = profile(active=[2])
    assert gallery['active_feature_count'] == 1
    assert scaled_manhattan_three(gallery, np.ones(11)) is None
    result = score_windows(np.ones((5, 11)), np.zeros(5, int), 1, {'a': gallery}, ['a'])
    assert not result['available'].any()
    assert np.isnan(result['distances']).all()
    assert result['active_feature_counts'] == {'a': 0}


def test_partial_active_subset_ignores_zero_spread_and_preserves_profile():
    gallery = profile(active=[1, 2]); before = copy.deepcopy(gallery)
    vector = np.full(11, 1e6); vector[1] = 3.
    assert scaled_manhattan_three(gallery, vector) == 2.
    assert gallery == before


def test_window_mean_before_distance_five_boundary_and_missing_slots():
    gallery = profile()
    taps = np.repeat(np.array([0., 0., 1., 2., 2., 0., 0., 0., 0.])[:, None], 11, axis=1)
    result = score_windows(taps, np.r_[np.ones(5, int), np.full(4, 2)], 4,
                           {'a': gallery}, ['a', 'b'])
    np.testing.assert_array_equal(result['available'],
                                   [[False, False], [True, False], [False, False], [False, False]])
    assert result['distances'][1, 0] == 0.
    assert np.isnan(result['distances'][~result['available']]).all()
    assert result['tap_counts'].tolist() == [0, 5, 4, 0]


def test_profile_with_less_than80_enrollment_taps_cannot_bypass_gallery_rule():
    taps = np.repeat(np.array([0., 2.])[:, None], 11, axis=1)
    gallery = fit_profile(taps, conventions=CONVENTIONS, min_taps=2)
    with pytest.raises(ValueError, match='minimum80'):
        scaled_manhattan_three(gallery, np.ones(11))
    with pytest.raises(ValueError, match='minimum80'):
        score_windows(np.empty((0, 11)), np.empty(0, int), 1, {'a': gallery}, ['a'])


@pytest.mark.parametrize('vector', [np.ones(3), np.full(11, np.nan), np.full(11, np.inf)])
def test_invalid_authentication_vector_rejected(vector):
    with pytest.raises(ValueError): scaled_manhattan_three(profile(), vector)


def test_missing_gallery_and_empty_windows_keep_all_slots():
    result = score_windows(np.empty((0, 11)), np.empty(0, int), 3, {}, ['a', 'b'])
    assert result['distances'].shape == (3, 2)
    assert not result['available'].any()
    assert result['active_feature_counts'] == {'a': 0, 'b': 0}
