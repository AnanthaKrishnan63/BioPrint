import copy

import pytest

from type2branch_length_completion import completed_pair


def fixture():
    arms = {}
    for arm in ('full', 'mixed'):
        records = [{'epoch': epoch, 'updates': (epoch + 1) * 100,
                    'mean_selection_frr_at_1pct': frr,
                    'mean_selection_eer': .1,
                    'checkpoint': f'{arm}/epoch_{epoch}'}
                   for epoch, frr in zip(range(2, 6), [.5, .3, .3, .4])]
        arms[arm] = {'status': 'paired_arm_complete', 'arm': arm, 'updates': 600,
                     'checkpoints': records, 'best_checkpoint': copy.deepcopy(records[1]),
                     'batch_stream_sha256': 'a' * 64}
    pair = {'status': 'paired_train_comparison_complete', 'selected_arm': 'full',
            'arm_best': {arm: copy.deepcopy(report['best_checkpoint']) for arm, report in arms.items()},
            'selected_checkpoint': copy.deepcopy(arms['full']['best_checkpoint']),
            'paired_batch_stream_sha256': 'a' * 64, 'max_warm_start_score_error': 0.}
    return pair, arms


def test_complete_earliest_checkpoint_and_full_arm_tie():
    pair, arms = fixture()
    assert completed_pair(pair, arms) == arms['full']['checkpoints'][1]


def test_mixed_can_win():
    pair, arms = fixture()
    chosen = arms['mixed']['checkpoints'][2]
    chosen['mean_selection_frr_at_1pct'] = .2
    arms['mixed']['best_checkpoint'] = copy.deepcopy(chosen)
    pair['arm_best']['mixed'] = copy.deepcopy(chosen)
    pair['selected_arm'] = 'mixed'
    pair['selected_checkpoint'] = copy.deepcopy(chosen)
    pair['max_warm_start_score_error'] = 1e-6
    assert completed_pair(pair, arms) == chosen


@pytest.mark.parametrize('field,value', [('status', 'running'), ('updates', 500),
    ('updates', 600.), ('updates', True), ('arm', 'wrong'),
    ('batch_stream_sha256', 'b' * 64), ('batch_stream_sha256', 'not-a-hash')])
def test_bad_arm_summary(field, value):
    pair, arms = fixture()
    arms['mixed'][field] = value
    with pytest.raises(ValueError):
        completed_pair(pair, arms)


@pytest.mark.parametrize('fault', ['missing', 'order', 'epoch_type', 'updates', 'path',
                                 'loss_nan', 'loss_range', 'wrong_best', 'later_tie'])
def test_bad_checkpoints(fault):
    pair, arms = fixture()
    arm = arms['full']
    records = arm['checkpoints']
    if fault == 'missing': records.pop()
    if fault == 'order': records.reverse()
    if fault == 'epoch_type': records[0]['epoch'] = 2.
    if fault == 'updates': records[1]['updates'] = 401
    if fault == 'path': records[0]['checkpoint'] = None
    if fault == 'loss_nan': records[0]['mean_selection_eer'] = float('nan')
    if fault == 'loss_range': records[0]['mean_selection_frr_at_1pct'] = 1.01
    if fault == 'wrong_best': arm['best_checkpoint'] = records[0]
    if fault == 'later_tie': arm['best_checkpoint'] = records[2]
    with pytest.raises(ValueError):
        completed_pair(pair, arms)


@pytest.mark.parametrize('field,value', [('status', 'running'), ('selected_arm', 'mixed'),
    ('selected_checkpoint', {}), ('arm_best', {}),
    ('paired_batch_stream_sha256', 'b' * 64),
    ('max_warm_start_score_error', -1e-12), ('max_warm_start_score_error', 1.000001e-6),
    ('max_warm_start_score_error', float('nan')), ('max_warm_start_score_error', float('inf')),
    ('max_warm_start_score_error', True), ('max_warm_start_score_error', '0')])
def test_bad_pair_summary(field, value):
    pair, arms = fixture()
    pair[field] = value
    with pytest.raises(ValueError):
        completed_pair(pair, arms)


@pytest.mark.parametrize('which,value', [('pair', None), ('pair', []), ('arms', None),
                                       ('arms', []), ('arms', {}), ('mixed', None)])
def test_malformed_reports(which, value):
    pair, arms = fixture()
    if which == 'pair': pair = value
    elif which == 'arms': arms = value
    else: arms[which] = value
    with pytest.raises(ValueError):
        completed_pair(pair, arms)


def test_no_mutation():
    pair, arms = fixture()
    before = copy.deepcopy((pair, arms))
    completed_pair(pair, arms)
    assert (pair, arms) == before
