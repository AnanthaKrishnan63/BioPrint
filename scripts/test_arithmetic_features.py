import copy
import numpy as np
import pytest
from arithmetic_features import parse_block, profile


def block(padded=False):
    data = {'imageT': [10., 13., 16., 19., 22.],
            'keyT': [11., [], 17.5, 21., 22.5], 'CorrectAns': [1, 0, 0, 1, 1]}
    if padded:
        data['imageT'].append([]); data['keyT'].append([]); data['CorrectAns'].append(0)
    return data


def test_padding_is_not_a_failed_trial_and_missing_is_not_observed_rt():
    clean = parse_block(block())
    assert parse_block(block(True)) == clean
    assert clean['response_time_seconds'][1] is None
    assert clean['bounded_wait_seconds'][1] == 2.5
    assert sum(clean['missing_response']) == 1 and len(clean['correct']) == 5


def test_all_timeouts_retained_as_censoring():
    data = block(); data['keyT'] = [[] for _ in range(5)]; data['CorrectAns'] = [0] * 5
    parsed = parse_block(data)
    assert parsed['response_time_seconds'] == [None] * 5
    assert parsed['bounded_wait_seconds'] == [2.5] * 5


@pytest.mark.parametrize('change', ['late', 'padding', 'missing_onset', 'correct_miss', 'unordered'])
def test_ambiguous_or_invalid_slots_rejected(change):
    data = block(True)
    if change == 'late': data['keyT'][0] = 12.6
    if change == 'padding': data['keyT'][5] = 99.
    if change == 'missing_onset': data['imageT'][2] = []
    if change == 'correct_miss': data['CorrectAns'][1] = 1
    if change == 'unordered': data['imageT'][1] = 9.
    with pytest.raises(ValueError): parse_block(data)


def test_slope_alone_discards_middle_level_information():
    levels = {key: block() for key in 'lmh'}
    original = profile(levels)
    modified = copy.deepcopy(levels); modified['m']['keyT'][0] = 12.
    changed = profile(modified)
    assert original['ordinal_wait_slope'] == changed['ordinal_wait_slope']
    assert original['condition_bounded_wait'] != changed['condition_bounded_wait']
    np.testing.assert_allclose(original['wait_missing_error_profile'][3:6], [.2] * 3)
