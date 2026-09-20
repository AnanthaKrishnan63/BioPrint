import copy
import pytest
from build_research_summary import paired_type2branch_rows


def generated():
    def metrics(n, fr, fa):
        return {'genuine_comparisons': n, 'impostor_comparisons': n * 2,
                'eer_diagnostic': .4, 'operating_points': {'far_1pct': {
                    'false_rejections': fr, 'false_acceptances': fa,
                    'frr': fr / n, 'far': fa / (2 * n)}}}
    methods = ['typenet', 'sapimouse', 'handcrafted_behavior', 'old_dual_neural',
               'old_hybrid', 'type2branch', 'type2branch_pointer', 'type2branch_hybrid']
    value = {'pooled': metrics(81, 54, 27), 'by_probe_identity': {
        subject: metrics(27, 18, 9) for subject in ['a','b','c']}}
    return {'status': 'paired_dev_evaluation_complete', 'evaluation_split': 'dev',
            'test_observations_decoded': False, 'prior_dev_exposure': True, 'model_updates': 0,
            'subjects': ['a','b','c'], 'claims': 243, 'windows': 140, 'below25_windows': 81,
            'results': {name: copy.deepcopy(value) for name in methods}}


def test_eight_rows_preserve_counts_and_limitations():
    rows = paired_type2branch_rows(generated(), 'generated.json')
    assert len(rows) == 8
    for row in rows:
        assert row['genuine_n'] == 81 and row['impostor_n'] == 162
        assert row['false_acceptances'] == 27 and row['false_rejections'] == 54
        assert row['far'] == 1/6 and row['frr'] == 2/3
        assert row['paired_windows'] == 140 and row['windows_below_source_training_minimum25'] == 81
        assert row['prior_dev_exposure'] and 'offline' in row['api_boundary']


@pytest.mark.parametrize('defect', ['test','unfinished','exposure','fit','missing_model','missing_person',
                                  'count','rate','nan','short','fractional_count','per_person_count'])
def test_rejects_incomplete_or_inconsistent_evidence(defect):
    report = generated(); pooled = report['results']['typenet']['pooled']
    if defect == 'test': report['test_observations_decoded'] = True
    elif defect == 'unfinished': report['status'] = 'running'
    elif defect == 'exposure': report['prior_dev_exposure'] = False
    elif defect == 'fit': report['model_updates'] = 1
    elif defect == 'missing_model': del report['results']['typenet']
    elif defect == 'missing_person': del report['results']['typenet']['by_probe_identity']['a']
    elif defect == 'count': report['claims'] += 1
    elif defect == 'rate': pooled['operating_points']['far_1pct']['far'] = .1
    elif defect == 'nan': pooled['eer_diagnostic'] = float('nan')
    elif defect == 'short': report['below25_windows'] = 141
    elif defect == 'fractional_count': pooled['genuine_comparisons'] = 81.
    elif defect == 'per_person_count':
        part = report['results']['typenet']['by_probe_identity']['a']['operating_points']['far_1pct']
        part['false_rejections'] = 0; part['frr'] = 0.
    with pytest.raises(ValueError): paired_type2branch_rows(report, 'generated.json')
