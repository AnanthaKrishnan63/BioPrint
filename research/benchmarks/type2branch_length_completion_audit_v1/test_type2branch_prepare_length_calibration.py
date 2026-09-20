import numpy as np
import pytest
from type2branch_prepare_length_calibration import raw_prefix_window, ordered_indices


@pytest.mark.parametrize('length', [25, 50, 75, 100])
def test_raw_prefix_preserves_signed_timing_and_normalizes_base(length):
    raw = np.tile([65, 100, -50], (100, 1))
    original = raw.copy()
    window = raw_prefix_window(raw, length)
    assert window['true_length'] == length
    np.testing.assert_array_equal(window['raw_ms'], raw[:length])
    np.testing.assert_array_equal(window['base'][:, 2], 0)
    np.testing.assert_array_equal(window['base'][:, 1], .1)
    np.testing.assert_array_equal(raw, original)


def test_no_implicit_padding_or_unrecognized_length():
    for raw, length in [(np.zeros((24, 3)), 25), (np.zeros((100, 3)), 38),
                        (np.zeros((100, 3)), True)]:
        with pytest.raises(ValueError): raw_prefix_window(raw, length)


def test_metadata_order_and_missing_or_duplicate_windows():
    allowed = [f'generated{i:02d}' for i in range(16)]
    subjects = np.repeat(allowed, 15); windows = np.tile(np.arange(15), 16)
    order = ordered_indices(subjects[::-1], windows[::-1], allowed)
    np.testing.assert_array_equal(subjects[::-1][order], subjects)
    np.testing.assert_array_equal(windows[::-1][order], windows)
    windows[1] = 0
    with pytest.raises(ValueError): ordered_indices(subjects, windows, allowed)
    with pytest.raises(ValueError): ordered_indices(subjects[:-1], windows[:-1], allowed)
    with pytest.raises(ValueError): ordered_indices(subjects, windows, allowed[:-1])
