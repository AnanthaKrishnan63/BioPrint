import copy

import pytest

from build_research_summary import capture_summary_entry


def fixture():
    return {'status': 'frozen_capture_dev_evaluation_complete', 'sealed_test_payloads_decoded': False,
        'prior_exposure': ['Earlier DEV metadata inspected'], 'selected_checkpoint': {'epoch': 3, 'updates': 400},
        'metrics': {'coverage': {'eligible': 2, 'total': 3, 'fraction': 2/3, 'ineligible_count': 1,
            'ineligible': [{'identity': 'c', 'reason': 'short'}]},
            'whole_cohort': {'total': 3, 'direct_acceptances': 1, 'score_rejections': 1,
                'ineligible': 1, 'additional_verifications': 2, 'additional_verification_fraction': 2/3},
            'conditional_metrics': {'pooled': {'genuine_n': 2, 'impostor_n': 4, 'false_acceptances': 1,
                'false_rejections': 1, 'far': .25, 'frr': .5, 'discrete_eer': .375,
                'margin_discrete_eer': .25}}}}


def test_one_conditional_row_preserves_coverage_and_margin_diagnostic():
    result = fixture(); before = copy.deepcopy(result)
    row, unavailable = capture_summary_entry(result, 'generated/report.json')
    assert unavailable is None
    assert (row['eer'], row['far'], row['frr']) == (.375, .25, .5)
    assert row['margin_eer_diagnostic_additional'] == .25
    assert row['genuine_probe_coverage'] == 2/3
    assert row['genuine_n'] == 2 and row['impostor_n'] == 4
    assert row['whole_cohort_additional_verification_fraction'] == 2/3
    assert 'Earlier DEV' in row['prior_dev_exposure']
    assert result == before


def test_no_eligible_captures_is_status_not_zero_metric_row():
    result = fixture()
    result['metrics']['coverage'] = {'eligible': 0, 'total': 3, 'fraction': 0., 'ineligible_count': 3,
        'ineligible': [{'identity': s, 'reason': 'short'} for s in ['a', 'b', 'c']]}
    result['metrics']['whole_cohort'] = {'total': 3, 'direct_acceptances': 0, 'score_rejections': 0,
        'ineligible': 3, 'additional_verifications': 3, 'additional_verification_fraction': 1.}
    result['metrics']['conditional_metrics'] = {}
    row, unavailable = capture_summary_entry(result, 'generated/report.json')
    assert row is None and unavailable['status'] == 'no_eligible_captures'
    assert not unavailable['recognition_metrics_available']
    assert 'far' not in unavailable and 'eer' not in unavailable


@pytest.mark.parametrize('fault', ['status', 'sealed', 'coverage', 'missing_count', 'reason',
    'whole', 'denominator', 'rate', 'nan_eer', 'bool_count', 'exposure'])
def test_inconsistent_report_rejected(fault):
    result = fixture(); metrics = result['metrics']
    if fault == 'status': result['status'] = 'running'
    if fault == 'sealed': result['sealed_test_payloads_decoded'] = True
    if fault == 'coverage': metrics['coverage']['fraction'] = 1.
    if fault == 'missing_count': metrics['coverage']['ineligible_count'] = 0
    if fault == 'reason': metrics['coverage']['ineligible'][0]['reason'] = ''
    if fault == 'whole': metrics['whole_cohort']['additional_verification_fraction'] = .5
    if fault == 'denominator': metrics['conditional_metrics']['pooled']['impostor_n'] = 3
    if fault == 'rate': metrics['conditional_metrics']['pooled']['far'] = 0.
    if fault == 'nan_eer': metrics['conditional_metrics']['pooled']['discrete_eer'] = float('nan')
    if fault == 'bool_count': metrics['coverage']['eligible'] = True
    if fault == 'exposure': result['prior_exposure'] = []
    with pytest.raises(ValueError): capture_summary_entry(result, 'generated/report.json')
