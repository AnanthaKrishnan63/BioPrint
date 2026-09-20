import numpy as np
import pytest
from pointer_sapimouse_znorm import fit_znorm, normalize, training_entries


def test_normalization_is_account_specific_and_probe_independent():
    mean, spread = fit_znorm([[1., 3.], [10., 14.]])
    np.testing.assert_array_equal(mean, [2., 12.])
    np.testing.assert_array_equal(spread, [1., 2.])
    result = normalize([[2., 4.], [14., 10.]], mean, spread)
    np.testing.assert_array_equal(result, [[0., 2.], [1., -1.]])
    np.testing.assert_array_equal(normalize([[2.], [14.]], mean, spread), result[:, :1])


def test_zero_background_spread_is_not_silently_repaired():
    with pytest.raises(ValueError): fit_znorm([[1., 1.]])


def test_only_designated_train_records_are_selected():
    files = []
    for i in range(60):
        files.append({'split': 'train', 'role': 'representation', 'path': f'x/bg{i}/x_3min.csv'})
    for i in range(12):
        for n in [1, 3]: files.append({'split': 'train', 'role': 'calibration', 'path': f'x/cal{i}/x_{n}min.csv'})
    files += [{'split': 'test_sealed', 'role': 'representation', 'path': 'never/test/x_3min.csv'},
              {'split': 'train', 'role': 'dev_support_enrollment', 'path': 'never/dev/x_3min.csv'}]
    result = training_entries({'files': files})
    assert [len(result[k]) for k in ['background', 'enrollment', 'calibration']] == [60, 12, 12]
    assert all('never' not in e['path'] for entries in result.values() for e in entries)


def test_path_escape_rejected_before_any_record_read():
    with pytest.raises(ValueError, match='escapes'):
        training_entries({'files': [{'split': 'train', 'role': 'representation', 'path': '../../x_3min.csv'}]})
