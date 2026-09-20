import copy

import pytest

from type2branch_capture_calibration_gate import calibrated_capture


def fixture():
    arms = {}
    for arm in ['full', 'mixed']:
        records = [{'epoch': epoch, 'updates': (epoch + 1) * 100,
                    'mean_selection_frr_at_1pct': .5, 'mean_selection_eer': .2,
                    'checkpoint': f'{arm}/{epoch}'} for epoch in range(2, 6)]
        arms[arm] = {'status': 'paired_arm_complete', 'arm': arm, 'updates': 600,
                     'checkpoints': records, 'best_checkpoint': copy.deepcopy(records[0]),
                     'batch_stream_sha256': 'a' * 64}
    chosen = copy.deepcopy(arms['full']['best_checkpoint'])
    pair = {'status': 'paired_train_comparison_complete', 'selected_arm': 'full',
            'selected_checkpoint': chosen,
            'arm_best': {arm: copy.deepcopy(value['best_checkpoint']) for arm, value in arms.items()},
            'paired_batch_stream_sha256': 'a' * 64, 'max_warm_start_score_error': 0.}
    plan = {'selected_arm': 'full', 'checkpoint': copy.deepcopy(chosen),
            'lengths': [25, 50, 75, 100], 'target_far': .01}
    report = {'status': 'paired_length_train_calibration_complete', 'selected_arm': 'full',
              'selected_checkpoint': copy.deepcopy(chosen), 'lengths': {}}
    for length in plan['lengths']:
        report['lengths'][str(length)] = {'threshold': -length / 100, 'scores_sha256': 'b' * 64,
            'metrics': {'pooled': {'genuine_n': 160, 'impostor_n': 2400,
                'false_acceptances': 24, 'false_rejections': 80,
                'far': .01, 'frr': .5, 'discrete_eer': .3}}}
    return pair, arms, report, plan


def test_returns_exact_length_thresholds_without_mutation():
    args = fixture()
    before = copy.deepcopy(args)
    assert calibrated_capture(*args) == {25: -.25, 50: -.5, 75: -.75, 100: -1.}
    assert args == before


def test_pair_must_be_complete_before_calibration():
    pair, arms, report, plan = fixture()
    arms['mixed']['status'] = 'running'
    with pytest.raises(ValueError):
        calibrated_capture(pair, arms, None, None)


@pytest.mark.parametrize('which,field,value', [
    ('report', 'status', 'running'), ('report', 'selected_arm', 'mixed'),
    ('plan', 'selected_arm', 'mixed'), ('report', 'selected_checkpoint', {}),
    ('plan', 'checkpoint', {}), ('plan', 'lengths', [25, 50, 75]),
    ('plan', 'lengths', [25., 50, 75, 100]), ('plan', 'lengths', [50, 25, 75, 100]),
    ('plan', 'target_far', .05), ('plan', 'target_far', True),
    ('plan', 'target_far', '.01'), ('report', 'lengths', {})])
def test_plan_report_binding(which, field, value):
    pair, arms, report, plan = fixture()
    (report if which == 'report' else plan)[field] = value
    with pytest.raises(ValueError):
        calibrated_capture(pair, arms, report, plan)


@pytest.mark.parametrize('field,value', [('threshold', True), ('threshold', '0'),
    ('threshold', float('nan')), ('threshold', float('inf')), ('threshold', 10**400),
    ('scores_sha256', 'wrong'), ('scores_sha256', 'B' * 64), ('metrics', None)])
def test_invalid_length_result(field, value):
    args = fixture()
    args[2]['lengths']['25'][field] = value
    with pytest.raises(ValueError):
        calibrated_capture(*args)


@pytest.mark.parametrize('field,value', [
    ('genuine_n', 159), ('genuine_n', 160.), ('impostor_n', 2399),
    ('false_acceptances', -1), ('false_acceptances', 25), ('false_acceptances', True),
    ('false_rejections', 161), ('false_rejections', 80.),
    ('far', .01000000001), ('far', float('nan')), ('frr', .4), ('frr', True),
    ('discrete_eer', -.01), ('discrete_eer', 1.01),
    ('discrete_eer', float('inf')), ('discrete_eer', False)])
def test_invalid_metrics(field, value):
    args = fixture()
    args[2]['lengths']['50']['metrics']['pooled'][field] = value
    with pytest.raises(ValueError):
        calibrated_capture(*args)


def test_zero_errors_and_rate_roundoff_tolerance():
    args = fixture()
    pooled = args[2]['lengths']['100']['metrics']['pooled']
    pooled.update(false_acceptances=0, false_rejections=0, far=0., frr=5e-13, discrete_eer=0.)
    assert calibrated_capture(*args)[100] == -1.


@pytest.mark.parametrize('index', [2, 3])
def test_nonobject_calibration(index):
    args = list(fixture())
    args[index] = None
    with pytest.raises(ValueError):
        calibrated_capture(*args)
