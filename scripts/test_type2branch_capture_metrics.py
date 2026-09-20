import copy
import json

import numpy as np
import pytest

from type2branch_capture_metrics import capture_metrics


def fixture():
    return [np.array([[1., 0., 0.], [.5, .5, 0.]]), ['a', 'b'], [25, 50],
            ['a', 'b', 'c'], {'a': {'action': 'score', 'used_length': 25},
                'b': {'action': 'score', 'used_length': 50},
                'c': {'action': 'additional_verification', 'used_length': 0, 'reason': 'short'}},
            {25: .5, 50: .75, 75: .5, 100: .5}, {'ab': ['a', 'b'], 'c': ['c']}]


def test_conditional_counts_and_whole_cohort_denominators():
    result = capture_metrics(*fixture())
    pooled = result['conditional_metrics']['pooled']
    assert (pooled['genuine_n'], pooled['impostor_n']) == (2, 4)
    assert pooled['false_rejections'] == 1 and pooled['false_acceptances'] == 0
    assert pooled['frr'] == .5 and pooled['far'] == 0
    coverage = result['coverage']
    assert (coverage['eligible'], coverage['total'], coverage['fraction']) == (2, 3, 2/3)
    assert coverage['ineligible'] == [{'identity': 'c', 'reason': 'short'}]
    assert coverage['ineligible_count'] == 1
    assert coverage['by_cohort']['ab'] == {'eligible': 2, 'total': 2, 'fraction': 1., 'ineligible': []}
    assert coverage['by_cohort']['c'] == {'eligible': 0, 'total': 1, 'fraction': 0.,
                                        'ineligible': [{'identity': 'c', 'reason': 'short'}]}
    assert coverage['by_length']['25'] == {'eligible': 1, 'total': 3, 'fraction_of_all_accounts': 1/3}
    assert coverage['by_length']['75'] == {'eligible': 0, 'total': 3, 'fraction_of_all_accounts': 0.}
    whole = result['whole_cohort']
    assert whole['direct_acceptance_fraction'] == 1/3
    assert whole['score_rejection_fraction'] == whole['ineligible_fraction'] == 1/3
    assert whole['additional_verification_fraction'] == 2/3
    assert result['conditional_metrics']['within_cohort']['c'] == {}
    assert result['conditional_metrics']['within_cohort']['ab']['impostor_n'] == 2
    assert result['conditional_metrics']['probe_cohorts_full_claims']['ab']['impostor_n'] == 4
    assert result['conditional_metrics']['per_length']['75'] == {}
    json.dumps(result, allow_nan=False)


def test_length_thresholds_and_inclusive_ties():
    args = fixture()
    args[0][:] = .5
    result = capture_metrics(*args)['conditional_metrics']
    assert result['per_length']['25']['far'] == 1
    assert result['per_length']['25']['frr'] == 0
    assert result['per_length']['50']['far'] == 0
    assert result['per_length']['50']['frr'] == 1
    assert result['pooled']['discrete_eer'] == .5
    assert 'margin_discrete_eer' in result['pooled']


def test_float32_scores_do_not_round_threshold_down():
    args = fixture()
    args[0] = np.full((2, 3), .5, dtype=np.float32)
    args[5] = {length: float(np.nextafter(.5, np.inf)) for length in [25, 50, 75, 100]}
    assert capture_metrics(*args)['conditional_metrics']['pooled']['false_acceptances'] == 0


def test_all_ineligible_has_no_conditional_metrics():
    args = fixture()
    args[:3] = [np.empty((0, 3)), [], []]
    args[4] = {s: {'action': 'additional_verification', 'used_length': 0, 'reason': 'short'}
               for s in args[3]}
    result = capture_metrics(*args)
    assert result['conditional_metrics'] == {}
    assert result['coverage']['fraction'] == 0
    assert result['coverage']['ineligible_count'] == 3
    assert result['coverage']['by_cohort']['ab']['fraction'] == 0
    assert all(item['fraction_of_all_accounts'] == 0 for item in result['coverage']['by_length'].values())
    assert result['whole_cohort']['additional_verification_fraction'] == 1


@pytest.mark.parametrize('fault', ['duplicate_probe', 'missing_probe', 'extra_probe', 'length_mismatch',
    'float_length', 'bool_length', 'missing_coverage', 'bad_action', 'bad_ineligible_length',
    'missing_reason', 'duplicate_identity', 'nonfinite_score', 'wrong_shape', 'nan_threshold',
    'bool_threshold', 'missing_anchor', 'cohort_overlap', 'cohort_missing', 'duplicate_cohort_member'])
def test_invalid_contract(fault):
    args = fixture()
    if fault == 'duplicate_probe': args[1] = ['a', 'a']
    if fault == 'missing_probe': args[1] = ['a']
    if fault == 'extra_probe': args[1] = ['a', 'c']
    if fault == 'length_mismatch': args[2] = [25, 75]
    if fault == 'float_length': args[2] = [25., 50.]
    if fault == 'bool_length': args[2] = [True, 50]
    if fault == 'missing_coverage': del args[4]['c']
    if fault == 'bad_action': args[4]['a']['action'] = 'drop'
    if fault == 'bad_ineligible_length': args[4]['c']['used_length'] = 25
    if fault == 'missing_reason': del args[4]['c']['reason']
    if fault == 'duplicate_identity': args[3] = ['a', 'b', 'b']
    if fault == 'nonfinite_score': args[0][0, 0] = np.nan
    if fault == 'wrong_shape': args[0] = np.zeros((2, 2))
    if fault == 'nan_threshold': args[5][25] = np.nan
    if fault == 'bool_threshold': args[5][25] = True
    if fault == 'missing_anchor': del args[5][100]
    if fault == 'cohort_overlap': args[6]['c'] = ['a', 'c']
    if fault == 'cohort_missing': args[6]['c'] = []
    if fault == 'duplicate_cohort_member': args[6]['ab'] = ['a', 'a', 'b']
    with pytest.raises(ValueError): capture_metrics(*args)


def test_probe_order_is_respected():
    args = fixture()
    baseline = capture_metrics(*args)
    args[0] = args[0][::-1]
    args[1].reverse(); args[2].reverse()
    assert capture_metrics(*args) == baseline


def test_inputs_unchanged():
    args = fixture()
    before = copy.deepcopy(args)
    capture_metrics(*args)
    np.testing.assert_array_equal(args[0], before[0])
    assert args[1:] == before[1:]


def test_margin_eer_distinguishes_length_adjustment_from_raw_scores():
    scores = np.array([[10., 9.], [0., 1.]])
    coverage = {identity: {'action': 'score', 'used_length': length}
                for identity, length in [('a', 25), ('b', 50)]}
    result = capture_metrics(scores, ['a', 'b'], [25, 50], ['a', 'b'], coverage,
        {25: 9.5, 50: .5, 75: .5, 100: .5}, {'all': ['a', 'b']})
    assert result['conditional_metrics']['pooled']['discrete_eer'] == .5
    assert result['conditional_metrics']['pooled']['margin_discrete_eer'] == 0


def test_partially_covered_cohort_and_length_fractions_sum():
    args = fixture()
    args[6] = {'ac': ['a', 'c'], 'b': ['b']}
    coverage = capture_metrics(*args)['coverage']
    assert coverage['by_cohort']['ac']['fraction'] == .5
    assert coverage['by_cohort']['ac']['ineligible'] == [{'identity': 'c', 'reason': 'short'}]
    assert (sum(item['fraction_of_all_accounts'] for item in coverage['by_length'].values())
            + coverage['ineligible_count'] / coverage['total']) == 1
