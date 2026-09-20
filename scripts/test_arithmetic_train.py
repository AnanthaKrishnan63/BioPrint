import numpy as np
import pytest
from arithmetic_train import records, score, threshold, choose, fit_metric, labels


def test_train_loader_rejects_dev_and_test_before_io():
    for role in ['dev', 'test_sealed', 'test', 'training']:
        with pytest.raises(ValueError): records(role, {})


def test_score_axes_and_metric_match_hand_calculation():
    parameters = {'active': [True, True], 'spread': [2., 1.], 'precision': [[4., 0.], [0., 1.]]}
    gallery = np.array([[0., 0.], [2., 1.]])
    query = np.array([[2., 0.], [0., 2.], [4., 3.]])
    np.testing.assert_allclose(score('scaled_l1', parameters, gallery, query), -np.array([[.5, 1, 2.5], [.5, 1, 1.5]]))
    expected = -np.sqrt([[4, 4, 25], [1, 5, 8]])
    np.testing.assert_allclose(score('shrinkage_mahalanobis', parameters, gallery, query), expected)
    assert labels(['A', 'B'], 2).tolist() == [1, 1, 0, 0, 0, 0, 1, 1]


def test_low_far_threshold_excludes_ties_and_small_sample_errors():
    y = np.array([1, 0, 0, 0]); s = np.array([.9, .8, .8, .1])
    cutoff = threshold(y, s, .01)
    assert cutoff > .8 and not np.any(s[y == 0] >= cutoff)


def test_selection_preserves_failure_and_low_far_priority():
    results = {'unsafe': {'far': .02, 'frr': 0., 'eer': .1},
               'all_reject': {'far': 0., 'frr': 1., 'eer': .4}}
    assert choose(results) == ('all_reject', 'no_useful_discrimination')
    assert choose({'unsafe': results['unsafe']}) == ('unsafe', 'no_low_far_qualifier')


def test_fit_ignores_constant_dimensions_and_uses_support_only():
    support = np.array([[[0., 9.], [1., 9.]], [[3., 9.], [5., 9.]]])
    p = fit_metric(support)
    assert p['active'] == [True, False]
    assert np.asarray(p['precision']).shape == (1, 1)
    with pytest.raises(ValueError): fit_metric(np.ones((2, 2, 1)))
