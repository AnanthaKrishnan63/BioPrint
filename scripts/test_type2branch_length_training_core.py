import json
import random

import numpy as np
import pytest

from type2branch_length_training_core import augment_prefixes, selection_key, tuple_state


def test_prefix_oracle_all_channels_and_no_mutation():
    batch = np.arange(4 * 100 * 5, dtype=np.float32).reshape(4, 100, 5) + 1
    batch[:, :, 3:] *= -1
    before = batch.copy()
    result = augment_prefixes(batch, [25, 50, 75, 100])
    assert result.dtype == batch.dtype and not np.shares_memory(result, batch)
    np.testing.assert_array_equal(batch, before)
    for index, length in enumerate([25, 50, 75, 100]):
        np.testing.assert_array_equal(result[index, :length], batch[index, :length])
        np.testing.assert_array_equal(result[index, length:], 0)


def test_full_length_copy_and_readonly_input():
    batch = np.ones((1, 100, 5))
    batch.flags.writeable = False
    result = augment_prefixes(batch, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(result, batch)
    result[0, 0, 0] = 7
    assert batch[0, 0, 0] == 1


@pytest.mark.parametrize('shape', [(0, 100, 5), (100, 5), (1, 99, 5), (1, 100, 4)])
def test_batch_shape(shape):
    with pytest.raises(ValueError):
        augment_prefixes(np.zeros(shape), [25])


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_even_in_discarded_tail(value):
    batch = np.ones((1, 100, 5))
    batch[0, -1, -1] = value
    with pytest.raises(ValueError):
        augment_prefixes(batch, [25])


@pytest.mark.parametrize('lengths', [[], [25, 50], [25.], [True], ['25'],
                                     [0], [24], [26], [101], [[25]], None])
def test_bad_lengths(lengths):
    with pytest.raises(ValueError):
        augment_prefixes(np.ones((1, 100, 5)), lengths)


def test_mixed_bool_length_is_not_coerced_to_integer():
    with pytest.raises(ValueError):
        augment_prefixes(np.ones((2, 100, 5)), [25, True])


@pytest.mark.parametrize('dtype', [bool, complex, str, object])
def test_nonreal_or_coerced_batch(dtype):
    with pytest.raises(ValueError):
        augment_prefixes(np.ones((1, 100, 5)).astype(dtype), [25])


def record(frr=.2, eer=.1, updates=300):
    return {'mean_selection_frr_at_1pct': frr, 'mean_selection_eer': eer, 'updates': updates}


def test_selection_priority_and_earliest_update():
    records = [record(.2, .1, 400), record(.2, .1, 300), record(.2, .2, 100), record(.3, 0, 0)]
    assert min(records, key=selection_key) is records[1]
    assert selection_key(record(0, 1, 0)) == (0., 1., 0)


@pytest.mark.parametrize('field,value', [
    ('mean_selection_frr_at_1pct', np.nan), ('mean_selection_eer', np.inf),
    ('mean_selection_frr_at_1pct', -1e-9), ('mean_selection_eer', 1.01),
    ('mean_selection_eer', '0.1'), ('mean_selection_eer', True),
    ('updates', -1), ('updates', 300.), ('updates', True), ('updates', np.inf)])
def test_invalid_selection_record(field, value):
    candidate = record()
    candidate[field] = value
    with pytest.raises(ValueError):
        selection_key(candidate)


def test_missing_selection_fields():
    with pytest.raises(ValueError):
        selection_key({})


def test_json_random_state_restores_next_draws_and_gaussian_cache():
    original = random.Random(20260920)
    original.gauss(0, 1)
    serialized = json.loads(json.dumps(original.getstate()))
    restored = random.Random()
    restored.setstate(tuple_state(serialized))
    assert restored.gauss(0, 1) == original.gauss(0, 1)
    assert [restored.random() for _ in range(20)] == [original.random() for _ in range(20)]
    assert isinstance(serialized, list) and isinstance(serialized[1], list)
