import numpy as np
import pytest
from type2branch_residual_features import residual_features
from type2branch_synthesis_cleanup import INVALID_TIMING


def test_signed_residuals_missing_internal_and_zero_key():
    base = np.array([[0, .1, 0], [65/255, .2, .4], [66/255, .3, .1]])
    synthetic = [[0, 200, INVALID_TIMING], [65, 100, INVALID_TIMING], [66, 300, 200]]
    features, valid = residual_features(base, synthetic)
    np.testing.assert_array_equal(features[:, :3], base)
    np.testing.assert_allclose(features[:, 3:], [[-.1, 0], [.1, 0], [0, -.1]])
    np.testing.assert_array_equal(valid, [[True, False], [True, False], [True, True]])


@pytest.mark.parametrize('base,synthetic', [
    ([[65/255, .1, 0]], [[66, 100, 0]]),
    ([[65/255, np.nan, 0]], [[65, 100, 0]]),
    ([[65/255, 31, 0]], [[65, 100, 0]]),
    ([[65/255, .1, 0]], [[65, -1, 0]]),
    ([[65/255, .1, 0]], [[65, 1501, 0]]),
])
def test_invalid_inputs(base, synthetic):
    with pytest.raises(ValueError):
        residual_features(base, synthetic)
