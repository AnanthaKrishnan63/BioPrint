import numpy as np
import pytest
from engine import aligned_typing, scorer


def enrollment(n):
    names = []
    values = []
    for i in range(n):
        names.append(f'H.Key{i}#{i}')
        values.append(80. + i)
        if i + 1 < n:
            names += [f'DD.Pair{i}', f'UD.Pair{i}']
            values += [120. + i, 40.]
    return scorer.fit([values] * 10, names), np.array(values)


def test_arbitrary_password_schema_retains_personal_alignment():
    for n in (3, 8, 16, 32):
        model, values = enrollment(n)
        x = aligned_typing.comparison(model, [values, values * 1.5])
        assert x.shape == (2, 31)
        assert not x[0].any()
        assert x[1, -1] > 0
        np.testing.assert_allclose(x[1, -1], scorer.distance(model, values * 1.5)[0] / model.threshold)
        # Renaming physical keys does not change the shared matcher's inputs.
        model.names = [name.split('.')[0] + f'.Other{i}' for i, name in enumerate(model.names)]
        np.testing.assert_array_equal(x, aligned_typing.comparison(model, [values, values * 1.5]))


def test_mismatched_personal_schema_rejected():
    model, values = enrollment(5)
    with pytest.raises(ValueError):
        aligned_typing.comparison(model, [values[:-1]])


def test_equal_distribution_can_still_have_wrong_positional_rhythm():
    model, values = enrollment(5)
    changed = values.copy()
    changed[[0, 3]] = changed[[3, 0]]
    assert np.array_equal(np.sort(values), np.sort(changed))
    assert aligned_typing.comparison(model, [changed])[0, -1] > 0
