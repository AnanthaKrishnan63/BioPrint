"""Generated embeddings/tensors only; no participant data or trained weights."""
import numpy as np
import pytest
from hmog_verification import (mean_gallery, euclidean_scores, pooled_eer,
                               far_threshold, verification_rates, branch_embeddings)


def test_gallery_means_all_rows_without_normalizing_and_scores_sorted_accounts():
    x = np.zeros((3, 64)); x[:, 0] = [2, 4, 10]
    gallery = mean_gallery(x, ['b', 'b', 'a'])
    assert gallery['b'][0] == 3
    names, scores = euclidean_scores(np.zeros((1, 64)), gallery)
    assert names == ['a', 'b']
    np.testing.assert_array_equal(scores, [[10, 3]])
    assert euclidean_scores(np.zeros((2, 64)), {})[1].shape == (2, 0)
    assert mean_gallery(np.empty((0, 64)), np.array([], dtype=str)) == {}


@pytest.mark.parametrize('g,i,expected', [
    ([0, 1], [2, 3], 0.), ([2, 3], [0, 1], 1.),
    ([1, 1], [1, 1], .5), ([0, 1], [1, 1, 1], 1 / 3),
])
def test_interpolated_pooled_eer(g, i, expected):
    assert pooled_eer(g, i) == pytest.approx(expected)


@pytest.mark.parametrize('target', [0., .001, .01, .05, .25, .5, .75])
def test_threshold_ties_and_maximal_representability(target):
    impostor = np.array([0., 1., 1., 5.])
    threshold = far_threshold(impostor, target)
    assert np.mean(impostor <= threshold) <= target
    assert np.mean(impostor <= np.nextafter(threshold, np.inf)) > target


def test_threshold_can_accept_gap_genuine_without_accepting_forbidden_impostor():
    threshold = far_threshold([1., 3., 3., 8.], .25)
    assert 2.99 <= threshold < 3.
    assert far_threshold([0., 1.], 1.) == np.finfo(np.float64).max


def test_missing_coverage_kept_separate_from_observed_rates():
    result = verification_rates([1., 2.], [1., 3., 4.], 1.,
                                expected_genuine=4, expected_impostor=5)
    assert result['frr'] == .5
    assert result['far'] == 1 / 3
    assert result['missing_genuine'] == result['missing_impostor'] == 2
    assert result['genuine_coverage'] == .5
    assert result['genuine_not_accepted_rate'] == .75
    assert result['impostor_acceptances_per_expected_claim'] == .2
    absent = verification_rates([], [], 0., expected_genuine=2, expected_impostor=3)
    assert absent['far'] is absent['frr'] is None
    assert absent['genuine_not_accepted_rate'] == 1.


@pytest.mark.parametrize('bad', [[], [-1.], [np.nan], [np.inf], [[1.]]])
def test_eer_and_calibration_reject_invalid_classes(bad):
    with pytest.raises(ValueError):
        pooled_eer(bad, [1.])
    with pytest.raises(ValueError):
        far_threshold(bad, .01)


@pytest.mark.parametrize('expected', [0, -1, 1.5, True])
def test_coverage_denominator_cannot_hide_observed_claims(expected):
    with pytest.raises(ValueError):
        verification_rates([1.], [2.], 1., expected_genuine=expected)


def test_exact_author_branch_hooks_joint_parity_and_no_batchnorm_mutation():
    import torch
    from hmog_training_core import load_author_classes
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model_class, _ = load_author_classes()
    model = model_class(10, 24, 50, 100, 64).eval()
    key, imu = torch.randn(2, 50, 10), torch.randn(2, 100, 24)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        expected = model([key, imu])
    actual = branch_embeddings(model, key, imu)
    assert set(actual) == {'key', 'imu', 'joint'}
    assert torch.equal(actual['joint'], expected)
    with torch.inference_mode():
        assert torch.equal(model.linear_final(torch.cat([actual['key'], actual['imu']], -1)), expected)
    assert all(not x.requires_grad for x in actual.values())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert not model.linear_key._forward_hooks
    assert not model.linear_imu._forward_hooks
    assert not model.linear_final._forward_hooks
    model.train()
    with pytest.raises(ValueError, match='eval mode'):
        branch_embeddings(model, key, imu)
