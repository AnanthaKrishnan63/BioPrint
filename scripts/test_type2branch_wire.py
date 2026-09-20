import numpy as np
import pytest

from type2branch_wire import validate_features, validate_request


def windows(count=1):
    result = np.zeros((count, 100, 5), dtype=np.float64)
    result[:, :, 0] = 65 / 255
    result[:, :, 1] = .12
    result[:, 1:, 2] = .2
    result[:, :, 3:] = [-.1, .05]
    return result


def test_all_byte_codes_float64_and_float32():
    array = windows(3)
    array[:, :, 0] = np.resize(np.arange(256) / 255, (3, 100))
    for value in (array, array.astype(np.float32), array.tolist()):
        result = validate_features(value)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(np.rint(result[:, :, 0] * 255),
                                      np.rint(array[:, :, 0] * 255))


def test_boundaries_and_max_batch():
    array = windows(32)
    array[:, :, 1] = 30
    array[:, :, 3:] = [-1.5, 30]
    assert validate_features(array).shape == (32, 100, 5)


@pytest.mark.parametrize('shape', [(0, 100, 5), (33, 100, 5), (100, 5),
                                  (1, 99, 5), (1, 100, 4)])
def test_bad_shape(shape):
    with pytest.raises(ValueError):
        validate_features(np.zeros(shape))


@pytest.mark.parametrize('channel,value', [(0, -.01), (0, 1.01), (0, .12345),
    (1, -.001), (1, 30.000001), (2, -.001), (2, 30.000001),
    (3, -1.500001), (4, 30.000001), (3, np.nan), (4, np.inf), (4, 1e100)])
def test_invalid_channel_values(channel, value):
    array = windows()
    array[0, 2, channel] = value
    with pytest.raises(ValueError):
        validate_features(array)


def test_first_ft_rejected_before_float32_underflow():
    array = windows()
    array[0, 0, 2] = 1e-100
    with pytest.raises(ValueError):
        validate_features(array)


@pytest.mark.parametrize('value', [True, '0', None, 1j])
def test_no_scalar_coercion(value):
    array = windows().tolist()
    array[0][0][3] = value
    with pytest.raises(ValueError):
        validate_features(array)


def test_ragged_and_bool_array():
    array = windows().tolist()
    array[0][0].pop()
    for value in [array, np.zeros((1, 100, 5), dtype=bool)]:
        with pytest.raises(ValueError):
            validate_features(value)


def test_request_returns_id_and_float32_array():
    identifier, array = validate_request({'id': 0, 'features': windows().tolist()})
    assert identifier == 0 and array.dtype == np.float32


@pytest.mark.parametrize('identifier', [-1, True, 0., '1', None])
def test_invalid_request_id(identifier):
    with pytest.raises(ValueError):
        validate_request({'id': identifier, 'features': windows()})


@pytest.mark.parametrize('payload', [None, [], {}, {'id': 1},
    {'id': 1, 'features': [], 'path': '/somewhere'}])
def test_request_schema(payload):
    with pytest.raises(ValueError):
        validate_request(payload)
