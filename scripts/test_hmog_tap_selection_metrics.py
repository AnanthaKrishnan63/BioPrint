import numpy as np
import pytest
from hmog_tap_selection_metrics import evaluate, rank, summarize, METHOD_ORDER

THRESHOLDS = {'0.001': .5, '0.01': 1., '0.05': 2.}


def test_macro_weights_accounts_and_counts_missing_genuine():
    subjects = np.array(['a', 'b', 'b', 'b'])
    scores = np.array([[np.nan, np.nan], [3., .1], [3., .1], [3., .1]])
    result = evaluate(scores, np.isfinite(scores), subjects, ['a', 'b'], np.ones(4, dtype=bool), THRESHOLDS)
    assert result['macro_genuine_not_accepted'] == .5
    assert result['rates']['0.01']['genuine_not_accepted_rate'] == .25
    assert result['qualifies']
    assert result['rates']['0.01']['missing_impostor'] == 1


def test_false_accept_disqualifies_and_never_recalibrates():
    scores = np.array([[.1, .9], [3., .1]])
    result = evaluate(scores, np.ones_like(scores, dtype=bool), ['a', 'b'], ['a', 'b'], np.ones(2, dtype=bool), THRESHOLDS)
    assert not result['qualifies']
    assert result['rates']['0.01']['far'] == .5
    assert THRESHOLDS['0.01'] == 1.


def test_no_observed_impostors_cannot_qualify():
    scores = np.full((2, 2), np.nan)
    result = evaluate(scores, np.isfinite(scores), ['a', 'b'], ['a', 'b'], np.ones(2, dtype=bool), THRESHOLDS)
    assert not result['qualifies']
    assert result['eer'] is None
    assert result['macro_genuine_not_accepted'] == 1


def test_deterministic_tie_and_all_reject_status():
    reports = {name: {'qualifies': True, 'macro_genuine_not_accepted': 1., 'accepted_genuine': 0}
               for name in reversed(METHOD_ORDER)}
    result = rank(reports)
    assert result['selected_method'] == 'joint_tap11'
    assert result['status'] == 'no_useful_discrimination'
    reports['tap11'].update(macro_genuine_not_accepted=.5, accepted_genuine=1)
    assert rank(reports)['selected_method'] == 'tap11'
    for report in reports.values(): report['qualifies'] = False
    assert rank(reports)['status'] == 'infeasible'


def test_missing_prespecified_method_rejected():
    with pytest.raises(ValueError): rank({})


def test_shared_intersection_retains_original_denominators():
    scores = {name: np.array([[.1, 3.], [3., .1]]) for name in METHOD_ORDER}
    masks = {name: np.ones((2, 2), dtype=bool) for name in METHOD_ORDER}
    masks['tap11'][0] = False
    scores['tap11'][0] = np.nan
    result = summarize(scores, masks, ['a', 'b'], ['a', 'b'], np.ones(2, dtype=bool),
                       {name: THRESHOLDS for name in METHOD_ORDER})
    assert result['selected_method'] == 'joint_tap11'
    assert result['metrics']['key']['base_coverage']['macro_genuine_not_accepted'] == 0
    assert result['metrics']['key']['shared_intersection']['macro_genuine_not_accepted'] == .5
    assert result['metrics']['key']['shared_intersection']['rates']['0.01']['expected_genuine'] == 2
