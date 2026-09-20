import numpy as np
import pytest
from pointer_znorm_uncertainty import weighted_rates


def test_cluster_weights_equal_explicit_replication():
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([.9, .9, .1, .7, .6, .4])
    weights = np.array([2, 2, 0, 0, 3, 3])
    expected = weighted_rates(np.repeat(labels, weights), np.repeat(scores, weights),
                              np.ones(weights.sum()), .6)
    np.testing.assert_allclose(weighted_rates(labels, scores, weights, .6), expected)
    assert expected[0] == 2 / 5
    assert expected[1] == 0


def test_reject_missing_class_and_negative_weight():
    with pytest.raises(ValueError):
        weighted_rates([1, 0], [.8, .2], [1, 0], .5)
    with pytest.raises(ValueError):
        weighted_rates([1, 0], [.8, .2], [1, -1], .5)
