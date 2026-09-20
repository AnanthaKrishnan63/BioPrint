import numpy as np
import pytest

from type2branch_capture_wire import validate_request


def payload(lengths=(25, 50, 75, 100)):
    features = np.zeros((len(lengths), 100, 5), dtype=np.float64)
    for index, length in enumerate(lengths):
        features[index, :length, 0] = 65 / 255
        features[index, :length, 1] = .1
        features[index, 1:length, 2] = .25
        features[index, :length, 3:] = [-.05, .01]
    return {'id': 7, 'features': features.tolist(), 'true_lengths': list(lengths)}


def test_mixed_lengths_and_return_types():
    value = payload()
    identifier, features, lengths = validate_request(value)
    assert identifier == 7 and features.dtype == np.float32 and lengths.dtype == np.int64
    np.testing.assert_array_equal(lengths, [25, 50, 75, 100])
    assert features.shape == (4, 100, 5)


def test_real_zero_events_do_not_change_explicit_length():
    value = payload((25,))
    value['features'] = np.zeros((1, 100, 5)).tolist()
    _, _, lengths = validate_request(value)
    assert lengths.tolist() == [25]
    value['true_lengths'] = [100]
    assert validate_request(value)[2].tolist() == [100]


def test_explicit_all100_has_no_padding_requirement():
    value = payload((100, 100))
    assert validate_request(value)[2].tolist() == [100, 100]


@pytest.mark.parametrize('channel', range(5))
def test_nonzero_tail_every_channel(channel):
    value = payload((25,))
    value['features'][0][25][channel] = 1 / 255 if channel == 0 else .01
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('tiny', [1e-100, -1e-100])
def test_tail_checked_before_float32_underflow(tiny):
    value = payload((25,))
    value['features'][0][99][4] = tiny
    assert np.float32(tiny) == 0
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('lengths', [[], [25, 50], [True], [25.], ['25'], [0],
                                     [24], [26], [101], None, (25,)])
def test_invalid_explicit_lengths(lengths):
    value = payload((25,))
    value['true_lengths'] = lengths
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('identifier', [-1, True, 7., '7', None])
def test_invalid_id(identifier):
    value = payload((25,))
    value['id'] = identifier
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('fault', ['missing_length', 'extra', 'missing_id', 'nonobject'])
def test_exact_schema(fault):
    value = payload((25,))
    if fault == 'missing_length': del value['true_lengths']
    if fault == 'extra': value['path'] = '/arbitrary'
    if fault == 'missing_id': del value['id']
    if fault == 'nonobject': value = []
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('bad', [True, '0', np.nan, np.inf, -1.6])
def test_original_feature_rules_are_retained(bad):
    value = payload((25,))
    value['features'][0][1][3] = bad
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize('count', [0, 33])
def test_original_batch_bounds(count):
    value = {'id': 0, 'features': np.zeros((count, 100, 5)), 'true_lengths': [100] * count}
    with pytest.raises(ValueError):
        validate_request(value)
