import numpy as np
import pytest

from type2branch_csv_bridge import encode_csv, decode_csv, raw_residual_seconds, pad_features
from type2branch_input_audit import integer_csv_compatibility


def test_integer_wire_roundtrip_preserves_sentinel_and_extremes():
    rows = [[0, 0, -1], [255, 2**31 - 1, -(2**31)]]
    encoded = encode_csv(rows)
    assert integer_csv_compatibility(encoded) == {'accepted': 2, 'rejected': 0}
    np.testing.assert_array_equal(decode_csv(encoded, [0, 255]), rows)


@pytest.mark.parametrize('rows', [[], [[256, 1, 2]], [[65, .2, 3]],
                                [[65, np.nan, 0]], [[65, 2**31, 0]], [[65, 1]]])
def test_invalid_input_is_not_silently_quantized(rows):
    with pytest.raises(ValueError):
        encode_csv(rows)


@pytest.mark.parametrize('text', ['65,1,0\n', 'VK,HT,FT\n65,1.0,0\n',
                                 'VK,HT,FT\n65,1\n', 'VK,HT,FT\n\n',
                                 'VK,HT,FT\n65,1,0\n66,2,3\n'])
def test_malformed_or_extra_rows_fail(text):
    with pytest.raises(ValueError):
        decode_csv(text, [65])


def test_key_order_and_missing_events_fail():
    for keys in ([66, 65], [65, 66, 67]):
        with pytest.raises(ValueError):
            decode_csv(encode_csv([[65, 100, 0], [66, 120, 250]]), keys)


def test_residual_units_sign_and_padding():
    residual = raw_residual_seconds([[65, 120, 0], [66, 100, 250]],
                                    [[65, 100, -1], [66, 150, 300]])
    np.testing.assert_array_equal(residual, [[.02, .001], [-.05, -.05]])
    padded = pad_features(residual, length=100)
    assert padded.shape == (100, 2)
    np.testing.assert_array_equal(padded[:2], residual)
    assert not padded[2:].any()
    with pytest.raises(ValueError):
        pad_features(residual, length=1)
    with pytest.raises(ValueError):
        raw_residual_seconds([[65, 1, 0]], [[66, 1, 0]])
