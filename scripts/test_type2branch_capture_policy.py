import numpy as np
import pytest

import type2branch_capture_policy as policy

ROW = 'a,a,.1,.25,.35,.15,.25,'
TERMINAL = 'a,,.1,,,,,'


@pytest.mark.parametrize('count,expected', [(0, 0), (1, 0), (24, 0), (25, 25),
    (38, 25), (49, 25), (50, 50), (74, 50), (75, 75), (99, 75), (100, 100), (101, 100)])
def test_fixed_anchor_allocation(count, expected):
    rows = [ROW] * count
    original = rows.copy()
    window, audit = policy.allocate_capture(rows)
    assert rows == original
    assert audit['available_events'] == count
    assert audit['parsed_events'] == audit['used_length'] == expected
    assert audit['excluded_events'] == count - expected
    if expected:
        assert audit['action'] == 'score'
        assert window['true_length'] == expected
        assert window['raw_ms'].shape == (expected, 3)
        np.testing.assert_array_equal(window['raw_ms'][:, 1], 100)
        assert audit['adapter']['actual_session_end'] == (count == expected)
    else:
        assert window is None and audit['action'] == 'additional_verification'


@pytest.mark.parametrize('count', [0, 1, 24])
def test_short_capture_is_entirely_unparsed(monkeypatch, count):
    def forbidden(*args, **kwargs):
        raise AssertionError('Short capture must not reach timing parser')
    monkeypatch.setattr(policy, 'adapt_variable', forbidden)
    window, audit = policy.allocate_capture(['not csv or timing'] * count)
    assert window is None and audit['parsed_events'] == 0


@pytest.mark.parametrize('count', [38, 49, 74, 99, 101, 1000])
def test_malformed_excluded_suffix_stays_opaque(count):
    anchor = max(value for value in policy.ANCHORS if value <= count)
    window, audit = policy.allocate_capture([ROW] * anchor + ['malformed suffix'] * (count - anchor))
    assert window['true_length'] == anchor
    assert audit['parsed_events'] == anchor
    assert not audit['adapter']['actual_session_end']


def test_actual_terminal_at_exact_anchor_accepted():
    window, audit = policy.allocate_capture([ROW] * 24 + [TERMINAL])
    assert window['true_length'] == 25 and audit['adapter']['terminal_events'] == 1


def test_prefix_cut_is_not_a_terminal_record():
    with pytest.raises(ValueError):
        policy.allocate_capture([ROW] * 24 + [TERMINAL, 'unparsed suffix'])


@pytest.mark.parametrize('bad', ['malformed timing', 'a,a,.1,.2,.35,.15,.25,',
                                  'a,a,.1,,,,,'])
def test_required_malformed_row_fails_without_replacement(bad):
    rows = [ROW] * 25
    rows[10] = bad
    with pytest.raises(ValueError):
        policy.allocate_capture(rows)


def test_existing_unknown_key_mapping_is_preserved_and_reported():
    rows = [ROW] * 25
    rows[10] = 'a,a,a,.1,.25,.35,.15,.25,'
    window, audit = policy.allocate_capture(rows)
    assert window['raw_ms'][10, 0] == 0
    assert audit['adapter']['unparseable_key_prefixes'] == 1
    assert audit['adapter']['unknown_codes'] == 1
    assert audit['adapter']['adjacencies_with_uncheckable_keys'] == 2


@pytest.mark.parametrize('value', [None, 'row', (ROW,), [None], [1], [True]])
def test_input_container_contract(value):
    with pytest.raises(ValueError):
        policy.allocate_capture(value)
