"""Synthetic CMU alias/dispatch regressions; no real measurements are loaded."""
import copy
import numpy as np
import pytest
from fastapi import HTTPException
import research_api as api


def synthetic_model(method, gamma=.1, threshold=.3):
    return {'method': method, 'names': ['synthetic.H'], 'center': [0.], 'spread': [1.],
            'cap': 2., 'sv': [[0.]], 'dual': [1.], 'intercept': 0.,
            'gamma': gamma, 'population_scale': [1.], 'calibration': {'far_1pct': threshold}}


@pytest.fixture
def artifacts(monkeypatch):
    base = {'sB': synthetic_model('baseline', threshold=.4), 'sA': synthetic_model('baseline')}
    global_model = {'sB': synthetic_model('svm-10-0.1'), 'sA': synthetic_model('svm-10-0.1')}
    account = {'sB': synthetic_model('svm-1-1', gamma=1., threshold=-.2),
               'sA': synthetic_model('svm-10-0.1', gamma=.1, threshold=-.7)}
    old = {'selected': 'svm-10-0.1', 'models': {'baseline': base, 'svm-10-0.1': global_model}}
    new = {'selected': 'account_selected', 'protocol': {'original_selected': 'svm-10-0.1'},
           'models': {**copy.deepcopy(old['models']), 'account_selected': account}}
    far = {'selected': 'far_selected', 'protocol': {'original_selected': 'svm-10-0.1'},
           'models': {**copy.deepcopy(new['models']), 'far_selected': {
               'sA': synthetic_model('svm-1-10', gamma=2., threshold=-.1),
               'sB': synthetic_model('svm-10-1', gamma=.5, threshold=-.4)}}}
    by_dataset = {'cmu': old, 'cmu-account-selection': new, 'cmu-far-selection': far}
    monkeypatch.setattr(api, 'cmu_artifact', lambda dataset='cmu': by_dataset[dataset])
    frozen_loader, fitted_loader = api.frozen, api.fitted_model
    frozen_loader.cache_clear(); fitted_loader.cache_clear()
    yield old, new
    frozen_loader.cache_clear(); fitted_loader.cache_clear()


def test_account_dispatch_preserves_each_subject_kernel_and_threshold(artifacts):
    response = api.score('cmu-account-selection', api.ScoreBody(features=[[2.]], model='train_selected'))
    assert response['subjects'] == ['sA', 'sB']
    np.testing.assert_allclose(response['scores'], [[np.exp(-.4), np.exp(-4)]], atol=1e-15)
    assert response['thresholds'] == [.7, .2]
    assert response['accepted'] == [[False, False]]


@pytest.mark.parametrize('alias', ['baseline', 'train_selected'])
def test_original_cmu_behavior_matches_corresponding_new_control(artifacts, alias):
    old = api.score('cmu', api.ScoreBody(features=[[0.], [3.]], model=alias))
    new_alias = 'global_reference' if alias == 'train_selected' else 'baseline'
    new = api.score('cmu-account-selection', api.ScoreBody(features=[[0.], [3.]], model=new_alias))
    for key in ['subjects', 'scores', 'thresholds', 'accepted', 'score_direction']:
        assert old[key] == new[key]


def test_global_reference_alias_not_added_to_original_cmu(artifacts):
    with pytest.raises(HTTPException) as error:
        api.fitted_model('cmu', 'global_reference')
    assert error.value.status_code == 404


@pytest.mark.parametrize('dataset', ['cmu', 'cmu-account-selection', 'cmu-far-selection'])
@pytest.mark.parametrize('value', [None, float('nan')])
def test_both_cmu_variants_refuse_missing_timing(artifacts, dataset, value):
    with pytest.raises(HTTPException) as error:
        api.score(dataset, api.ScoreBody(features=[[value]]))
    assert error.value.status_code == 422


@pytest.mark.parametrize('dataset', ['cmu', 'cmu-account-selection', 'cmu-far-selection'])
def test_extreme_finite_input_rejected_before_model_load(artifacts, monkeypatch, dataset):
    def forbidden(*args):
        raise AssertionError('Unrepresentable inputs must not reach model inference')
    monkeypatch.setattr(api, 'fitted_model', forbidden)
    with pytest.raises(HTTPException) as error:
        api.score(dataset, api.ScoreBody(features=[[1e308]]))
    assert error.value.status_code == 422


@pytest.mark.parametrize('dataset', ['cmu', 'cmu-account-selection', 'cmu-far-selection'])
@pytest.mark.parametrize('kind', ['samples', 'raw'])
def test_cmu_test_routes_refuse_before_any_loader(monkeypatch, dataset, kind):
    import eval.strict_cmu as strict
    def forbidden(*args):
        raise AssertionError('No measurement loader may run')
    monkeypatch.setattr(api, 'samples', forbidden)
    monkeypatch.setattr(strict, 'load_partition', forbidden)
    route = api.sample_page if kind == 'samples' else api.raw_page
    with pytest.raises(HTTPException) as error:
        route(dataset, 'test', offset=0, limit=1)
    assert error.value.status_code == 403


def test_far_selector_and_eer_reference_use_distinct_frozen_models(artifacts):
    body = api.ScoreBody(features=[[2.]], model='train_selected')
    selected = api.score('cmu-far-selection', body)
    reference = api.score('cmu-far-selection', api.ScoreBody(features=[[2.]], model='eer_reference'))
    previous = api.score('cmu-account-selection', body)
    np.testing.assert_allclose(selected['scores'], [[np.exp(-8), np.exp(-2)]], atol=1e-15)
    assert selected['thresholds'] == [.1, .4]
    for key in ['subjects', 'scores', 'thresholds', 'accepted']:
        assert reference[key] == previous[key]
    assert selected['scores'] != reference['scores']


@pytest.mark.parametrize('dataset', ['cmu', 'cmu-account-selection'])
def test_eer_reference_only_available_for_far_experiment(artifacts, dataset):
    with pytest.raises(HTTPException) as error:
        api.fitted_model(dataset, 'eer_reference')
    assert error.value.status_code == 404
