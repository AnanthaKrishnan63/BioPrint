"""Generated feature archives only; no participant measurements are read."""
import numpy as np
import pytest

from hmog_train import load_fit_arrays


def generated_arrays():
    return {
        'key': np.stack([np.full((50, 10), i, np.float32) for i in range(4)]),
        'imu': np.stack([np.full((100, 24), i + 10, np.float32) for i in range(4)]),
        'subject': np.array(['717868', '717868', '526319', '526319']),
        'session': np.array([1, 9, 8, 16], dtype=np.int64),
        'role': np.array(['train_enrollment', 'train_fit', 'train_enrollment', 'train_fit']),
    }


def save_archive(tmp_path, arrays):
    path = tmp_path / 'synthetic-features.npz'
    np.savez(path, **arrays)
    return path


def test_filters_enrollment_preserving_feature_metadata_correspondence(tmp_path):
    arrays = generated_arrays()
    actual = load_fit_arrays(save_archive(tmp_path, arrays))
    for result, name in zip(actual, ['key', 'imu', 'subject', 'session', 'role']):
        np.testing.assert_array_equal(result, arrays[name][[1, 3]])
    assert actual[0].dtype == actual[1].dtype == np.float32


@pytest.mark.parametrize('subject', ['556357', '180679', '962159', 'unknown'])
def test_rejects_nonfit_identity_even_in_filtered_enrollment(tmp_path, subject):
    arrays = generated_arrays()
    arrays['subject'] = arrays['subject'].astype('U32')
    arrays['subject'][0] = subject
    with pytest.raises(PermissionError, match='non-fit identities'):
        load_fit_arrays(save_archive(tmp_path, arrays))


@pytest.mark.parametrize('session,role', [
    (9, 'dev_probe'), (9, 'train_selection'), (9, 'train_calibration'),
    (9, 'train_enrollment'), (8, 'train_fit'), (17, 'sealed_test'),
    (0, 'train_enrollment'), (17, 'train_fit'),
])
def test_rejects_session_role_mismatch(tmp_path, session, role):
    arrays = generated_arrays()
    arrays['session'][1] = session
    arrays['role'] = arrays['role'].astype('U32')
    arrays['role'][1] = role
    with pytest.raises(PermissionError, match='unauthorized record roles'):
        load_fit_arrays(save_archive(tmp_path, arrays))


@pytest.mark.parametrize('name,dtype', [
    ('key', np.float64), ('key', np.int32), ('imu', np.float64),
    ('session', np.float64), ('session', np.bool_),
])
def test_rejects_implicit_dtype_conversion(tmp_path, name, dtype):
    arrays = generated_arrays()
    arrays[name] = arrays[name].astype(dtype)
    with pytest.raises(ValueError, match='shapes/dtypes/values'):
        load_fit_arrays(save_archive(tmp_path, arrays))


@pytest.mark.parametrize('name,shape', [
    ('key', (4, 49, 10)), ('imu', (4, 100, 23)), ('imu', (3, 100, 24)),
    ('subject', (4, 1)), ('session', (4, 1)), ('role', (4, 1)),
])
def test_rejects_corrupt_shapes(tmp_path, name, shape):
    arrays = generated_arrays()
    arrays[name] = np.resize(arrays[name], shape)
    with pytest.raises(ValueError, match='shapes/dtypes/values'):
        load_fit_arrays(save_archive(tmp_path, arrays))


@pytest.mark.parametrize('name,value', [('key', np.nan), ('imu', np.inf), ('imu', -np.inf)])
def test_rejects_nonfinite_values_even_in_filtered_enrollment(tmp_path, name, value):
    arrays = generated_arrays()
    arrays[name][0, 0, 0] = value
    with pytest.raises(ValueError, match='shapes/dtypes/values'):
        load_fit_arrays(save_archive(tmp_path, arrays))


def test_rejects_archive_without_population_fit_rows(tmp_path):
    arrays = generated_arrays()
    for name in arrays:
        arrays[name] = arrays[name][[0, 2]]
    with pytest.raises(ValueError, match='No population-fit windows'):
        load_fit_arrays(save_archive(tmp_path, arrays))


def test_pickle_backed_feature_arrays_are_never_enabled(tmp_path):
    arrays = generated_arrays()
    arrays['key'] = arrays['key'].astype(object)
    with pytest.raises(ValueError, match='allow_pickle=False'):
        load_fit_arrays(save_archive(tmp_path, arrays))
