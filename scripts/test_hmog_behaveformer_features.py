"""Generated-data tests only: no HMOG archives, fixtures or observations."""
import numpy as np
import pytest
from hmog_behaveformer_features import key_features, sensor_channels, imu_window_features


def test_key_formula_order_scaling_and_overlap():
    p = np.array([0., 20., 70., 120.])
    r = np.array([30., 50., 80., 160.])
    original = [v.copy() for v in [p, r]]
    result = key_features(p, r, [-5, 65, 66, 67], tokens=2)
    expected = np.array([[30, -10, 20, 20, 50, 40, 70, 50, 80],
                         [30, 20, 50, 30, 60, 70, 100, 110, 140]]) / 1000
    np.testing.assert_array_equal(result[:, :9], expected)
    np.testing.assert_array_equal(result[:, 9], [-5 / 255, 65 / 255])
    np.testing.assert_array_equal(p, original[0]); np.testing.assert_array_equal(r, original[1])


def test_key_requires_actual_lookahead_and_preserves_time_translation():
    p = np.arange(52.) * 100; r = p + 40; keys = np.full(52, 65)
    np.testing.assert_array_equal(key_features(p, r, keys), key_features(p + 1e12, r + 1e12, keys))
    with pytest.raises(ValueError, match='tokens\\+2'):
        key_features(p[:-1], r[:-1], keys[:-1])


@pytest.mark.parametrize('p,r,k', [([1, 0, 2], [2, 1, 3], [1, 2, 3]),
    ([0, 1, 2], [-1, 2, 3], [1, 2, 3]), ([0, 1, 2], [1, 2, 3], [1.5, 2, 3]),
    ([0, 1, float('nan')], [1, 2, 3], [1, 2, 3])])
def test_invalid_key_inputs_rejected(p, r, k):
    with pytest.raises(ValueError):
        key_features(p, r, k, tokens=1)


def test_sensor_linear_ramp_derivatives_and_constant_fft():
    xyz = np.column_stack((np.arange(4.), np.full(4, 2.), 3 * np.arange(4.)))
    result = sensor_channels(xyz)
    np.testing.assert_array_equal(result[:, :3], xyz)
    np.testing.assert_array_equal(result[:, 4], [8, 0, 0, 0])
    np.testing.assert_array_equal(result[:, 6:9], np.tile([1, 0, 3], (4, 1)))
    np.testing.assert_array_equal(result[:, 9:], np.zeros((4, 3)))
    # Analytic four-point DFT of [0,1,2,3]: [6,-2+2i,-2,-2-2i].
    np.testing.assert_allclose(result[:, 3], [6, np.sqrt(8), 2, np.sqrt(8)], atol=1e-14)


def test_half_open_bins_exact_source_scaling_and_different_sampling_rates():
    a_t = np.arange(4.); a = np.ones((4, 3)) * 10
    g_t = np.arange(0, 4, .5); g = np.ones((8, 3)) * 2
    result = imu_window_features(a_t, a, g_t, g, start_ms=0, end_ms=4, bins=2)
    assert result.shape == (2, 24)
    np.testing.assert_array_equal(result[:, :3], np.ones((2, 3)))
    np.testing.assert_array_equal(result[:, 12:15], np.full((2, 3), 2.))
    np.testing.assert_allclose(result[:, 3:6], [[.02]*3, [0]*3])
    np.testing.assert_allclose(result[:, 15:18], [[.004]*3, [0]*3])
    np.testing.assert_array_equal(result[:, 6:12], np.zeros((2, 6)))


def test_outside_window_cannot_influence_fft_or_gradients():
    t = np.arange(6.)
    xyz = np.column_stack((t, t*t, t+1))
    original = imu_window_features(t, xyz, t, xyz, start_ms=1, end_ms=5, bins=2)
    changed = xyz.copy(); changed[[0, 5]] = 1e9
    actual = imu_window_features(t, changed, t, changed, start_ms=1, end_ms=5, bins=2)
    np.testing.assert_array_equal(actual, original)


def test_exact_internal_boundary_goes_only_to_next_bin():
    t = np.array([0., .5, 1., 1.5]); xyz = np.column_stack((t, t, t))
    result = imu_window_features(t, xyz, t, xyz, start_ms=0, end_ms=2, bins=2)
    np.testing.assert_allclose(result[:, 0], [.025, .125])


def test_missing_bin_never_filled_and_reversed_clock_never_sorted():
    xyz = np.ones((4, 3))
    with pytest.raises(ValueError, match='support'):
        imu_window_features([0, .1, .2, .3], xyz, [0, 1, 2, 3], xyz, start_ms=0, end_ms=4, bins=2)
    with pytest.raises(ValueError, match='nondecreasing'):
        imu_window_features([0, 2, 1, 3], xyz, [0, 1, 2, 3], xyz, start_ms=0, end_ms=4, bins=2)


@pytest.mark.parametrize('xyz', [np.ones((2, 3)), np.ones((4, 2)), np.full((4, 3), np.nan)])
def test_insufficient_or_invalid_sensor_records_rejected(xyz):
    with pytest.raises(ValueError):
        sensor_channels(xyz)
