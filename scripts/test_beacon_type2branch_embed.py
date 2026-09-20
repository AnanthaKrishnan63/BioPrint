import numpy as np
import pytest

from beacon_type2branch_embed import validate_arrays


def arrays():
    x = np.zeros((4, 100, 5), dtype=np.float64)
    lengths = np.array([5, 24, 100, 25])
    for features, length in zip(x, lengths):
        features[:length] = .1
    return {'features': x, 'true_length': lengths,
            'subject': np.array(['P002', 'P002', 'P003', 'P003']),
            'role': np.array(['enrollment', 'probe', 'enrollment', 'probe']),
            'start': np.array([0., 0., 30., 0.])}


def test_retains_short_inputs_and_metadata_exactly():
    data = arrays()
    x, metadata = validate_arrays(data, ['P002', 'P003'])
    assert x.dtype == np.float32
    for key, value in metadata.items():
        np.testing.assert_array_equal(value, data[key])
    np.testing.assert_array_equal(metadata['true_length'], [5, 24, 100, 25])


@pytest.mark.parametrize('defect', ['nan', 'padding', 'length', 'float_length', 'missing_subject',
                                  'missing_role', 'duplicate', 'order', 'start', 'shape', 'string_role'])
def test_invalid_inputs_rejected(defect):
    data = arrays()
    if defect == 'nan': data['features'][0, 0, 1] = np.nan
    elif defect == 'padding': data['features'][0, 5, 1] = .1
    elif defect == 'length': data['true_length'][0] = 4
    elif defect == 'float_length': data['true_length'] = data['true_length'].astype(float)
    elif defect == 'missing_subject': data['subject'][2:] = 'P004'
    elif defect == 'missing_role': data['role'][1] = 'enrollment'
    elif defect == 'duplicate':
        data['role'][1] = 'enrollment'
        data['start'][1] = data['start'][0]
    elif defect == 'order':
        for key in data: data[key] = data[key][::-1]
    elif defect == 'start': data['start'][0] = np.inf
    elif defect == 'shape': data['features'] = data['features'][:, :99]
    elif defect == 'string_role': data['role'] = data['role'].astype('S')
    with pytest.raises(ValueError):
        validate_arrays(data, ['P002', 'P003'])
