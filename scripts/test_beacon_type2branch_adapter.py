import copy
from decimal import Decimal

import numpy as np
import pytest

from beacon_type2branch_adapter import adapt_window, milliseconds, parse_rows, synthesize_rows


def row(press, release=None, key='a', duration=None):
    release = press + 10 if release is None else release
    duration = release - press if duration is None else duration
    return {'Elapsed Start Time': str(Decimal(press) / 1000),
            'Elapsed Release Time': str(Decimal(release) / 1000),
            'Duration': str(Decimal(duration) / 1000), 'Key': key}


def parse(rows):
    return parse_rows(rows, key_mapper=lambda key: {'a': 65, 'b': 66, 'unknown': 0}[key])


def test_stable_onset_sort_and_elapsed_hold_ignore_duration():
    rows = [row(200, 240, 'b', 42), row(100, 130), row(100, 120, 'b'),
            row(300), row(400)]
    before = copy.deepcopy(rows)
    parsed = parse(rows)
    window, audit = adapt_window(parsed, 0)
    np.testing.assert_array_equal(window['raw_ms'],
                                  [[65, 30, 0], [66, 20, 0], [66, 40, 100],
                                   [65, 10, 100], [65, 10, 100]])
    assert parsed['audit']['release_order_onset_decreases'] == 1
    assert parsed['audit']['maximum_duration_disagreement_ms'] == 2
    assert audit['below_source_minimum_25']
    assert rows == before


def test_whole_hold_boundary_and_float_window_start():
    rows = [row(999, 1001), row(1000), row(2000), row(3000), row(4000),
            row(30990, 30999), row(30999, 31000), row(31000, 31000)]
    window, audit = adapt_window(parse(rows), 1.)
    assert audit['eligible_events'] == 5
    assert window['raw_ms'][0, 2] == 0
    assert window['raw_ms'][-1, 1] == 9
    # The predecessor retained this exact float comparison, without ms rounding.
    with pytest.raises(ValueError, match='baseline key minimum'):
        adapt_window(parse(rows), np.nextafter(1., 2.))


@pytest.mark.parametrize('length', [5, 24, 25, 99, 100, 101, 150])
def test_true_lengths_truncation_and_padding(length):
    class Population:
        def predict(self, keys):
            assert len(keys) == min(length, 100)
            return np.full((len(keys), 2), np.nan), np.full((len(keys), 2), -1)

    class Rng:
        calls = 0

        def next_double(self):
            self.calls += 1
            return .5

    rng = Rng()
    result, audit = synthesize_rows([row(i * 100) for i in range(length)], 0,
                                   Population(), rng, key_mapper=lambda _: 65)
    kept = min(length, 100)
    assert result['true_length'] == kept
    assert audit['window']['truncated_events'] == max(0, length - 100)
    assert rng.calls == 2 * kept
    assert result['features'].shape == (100, 5)
    np.testing.assert_array_equal(result['features'][kept:], 0)
    np.testing.assert_array_equal(result['context_orders'][kept:], -2)
    assert not result['residual_valid'][kept:].any()
    assert result['features'][0, 2] == 0
    assert result['features'][0, 4] == 0
    assert not result['residual_valid'][0, 1]


def test_unknown_key_is_real_event_and_inputs_not_mutated():
    parsed = parse([row(i * 100, key='unknown') for i in range(5)])
    before = parsed['events_ms'].copy()
    window, audit = adapt_window(parsed, 0)
    assert parsed['audit']['unknown_codes'] == audit['unknown_codes'] == 5
    assert window['true_length'] == 5
    np.testing.assert_array_equal(parsed['events_ms'], before)


@pytest.mark.parametrize('value', ['0.0001', 'NaN', 'Infinity', 'broken', None])
def test_inexact_or_invalid_milliseconds_rejected(value):
    with pytest.raises(ValueError):
        milliseconds(value)


@pytest.mark.parametrize('start', [True, None, '0', float('nan'), float('inf')])
def test_invalid_start_rejected(start):
    with pytest.raises(ValueError):
        adapt_window(parse([row(i * 100) for i in range(5)]), start)


def test_empty_negative_hold_and_too_short_rejected():
    with pytest.raises(ValueError, match='No observed keys'):
        parse([])
    with pytest.raises(ValueError, match='Negative hold'):
        parse([row(10, 9)])
    with pytest.raises(ValueError, match='baseline key minimum'):
        adapt_window(parse([row(i * 100) for i in range(4)]), 0)
