"""Generated tap features only; no participant observations or archives."""
import numpy as np
import pytest

from hmog_tap_benchmark import build_profiles, score_windows, validate_taps


def generated_taps(n):
    return np.tile((np.arange(n) % 2)[:, None] * 2., (1, 11))


def test_enrollment_mask_excludes_same_identity_probe_values():
    taps = np.concatenate((generated_taps(80), np.full((20, 11), 100.)))
    indices = np.r_[np.zeros(80, int), np.ones(20, int)]
    profiles, diagnostics = build_profiles(taps, indices, ['a', 'a'], [True, False], ['a'])
    assert profiles['a']['mean'] == [1.] * 11
    assert profiles['a']['spread'] == [1.] * 11
    assert diagnostics['a']['enrollment_taps'] == 80


def test_minimum_gallery_support_and_missing_window_are_not_imputed():
    taps = np.concatenate((generated_taps(80), generated_taps(79), np.full((5, 11), 2.), np.ones((4, 11))))
    index = np.repeat(np.arange(4), [80, 79, 5, 4])
    profiles, diagnostics = build_profiles(taps, index, ['a', 'b', 'a', 'b'],
                                            [True, True, False, False], ['a', 'b'])
    assert set(profiles) == {'a'}
    assert diagnostics['b']['missing_reason'] == 'fewer_than_80_enrollment_taps'
    result = score_windows(taps, index, 4, profiles, ['a', 'b'])
    assert result['distances'][2, 0] == 11.
    assert result['available'][2].tolist() == [True, False]
    assert not result['available'][3].any()
    assert np.isnan(result['distances'][3]).all()


def test_constant_gallery_is_unavailable_instead_of_perfect_match():
    profiles, diagnostics = build_profiles(np.ones((80, 11)), np.zeros(80, int), ['a'], [True], ['a'])
    assert not profiles
    assert 'usable spread' in diagnostics['a']['missing_reason']


@pytest.mark.parametrize('index', [np.array([-1]), np.array([2]), np.array([0.]), np.array([True])])
def test_invalid_window_mapping_denied(index):
    with pytest.raises(ValueError):
        validate_taps(np.ones((1, 11)), index, 2)


def test_empty_tap_archive_preserves_missing_windows():
    result = score_windows(np.empty((0, 11)), np.empty(0, dtype=int), 2, {}, ['a'])
    assert not result['available'].any()
    assert result['tap_counts'].tolist() == [0, 0]


def test_enrollment_uses_individual_tap_weights_not_window_mean_weights():
    taps = np.r_[np.zeros((75, 11)), np.full((5, 11), 8.)]
    index = np.r_[np.zeros(75, int), np.ones(5, int)]
    profiles, _ = build_profiles(taps, index, ['a', 'a'], [True, True], ['a'])
    np.testing.assert_allclose(profiles['a']['mean'], np.full(11, .5))
    np.testing.assert_allclose(profiles['a']['spread'], np.full(11, np.sqrt(3.75)))
    assert profiles['a']['n_enrollment'] == 80


def test_scan_averages_vectors_before_absolute_standardized_distance():
    profiles, _ = build_profiles(generated_taps(80), np.zeros(80, int),
                                 ['a'], [True], ['a'])
    # Per-tap distances average to8.8, but the scan mean equals the template.
    probes = np.repeat(np.array([0., 0., 1., 2., 2.])[:, None], 11, axis=1)
    result = score_windows(probes, np.zeros(5, int), 1, profiles, ['a'])
    assert result['distances'][0, 0] == 0.


def test_zero_spread_features_remain_ignored_without_changing_masks():
    taps = np.ones((80, 11)); taps[:, 0] = np.arange(80) % 2 * 2
    profiles, _ = build_profiles(taps, np.zeros(80, int), ['a'], [True], ['a'])
    assert profiles['a']['active_feature_count'] == 1
    probes = np.full((5, 11), 1e6); probes[:, 0] = 3.
    result = score_windows(probes, np.ones(5, int), 3, profiles, ['a', 'b'])
    np.testing.assert_array_equal(result['available'], [[False, False], [True, False], [False, False]])
    assert result['distances'][1, 0] == 2.
    assert np.isnan(result['distances'][~result['available']]).all()


def test_eighty_taps_is_minimum_not_cap_and_mask_never_refits():
    taps = np.r_[generated_taps(80), np.full((20, 11), 5.), np.full((10, 11), 1000.)]
    index = np.repeat(np.arange(3), [80, 20, 10])
    profiles, _ = build_profiles(taps, index, ['a'] * 3, [True, True, False], ['a'])
    assert profiles['a']['n_enrollment'] == 100
    np.testing.assert_allclose(profiles['a']['mean'], np.full(11, 1.8))


def test_subfive_windows_do_not_supply_hidden_gallery_taps():
    taps = generated_taps(83)
    index = np.repeat(np.arange(2), [79, 4])
    profiles, diagnostics = build_profiles(taps, index, ['a', 'a'], [True, True], ['a'])
    assert profiles == {}
    assert diagnostics['a']['enrollment_taps'] == 79
    assert diagnostics['a']['enrollment_windows'] == 1
