import numpy as np
import pytest
from engine import supervised


def test_serialization_calibration_and_no_implicit_decision():
    rng = np.random.default_rng(7)
    positive = rng.normal(0, 1, (20, 3))
    negative = rng.normal(5, 1, (40, 3))
    model = supervised.fit(positive, negative, ['a', 'b', 'c'])
    with pytest.raises(ValueError, match='Calibrate'):
        supervised.accepts(model, positive)
    calibration = rng.normal(5, 1, (20, 3))
    supervised.calibrate(model, calibration, target_far=.1)
    assert supervised.accepts(model, calibration).mean() <= .1
    restored = supervised.Profile.from_dict(model.to_dict())
    np.testing.assert_allclose(supervised.distances(model, positive), supervised.distances(restored, positive))
    assert supervised.accepts(restored, positive).mean() > .9


def test_invalid_schema_and_nan():
    with pytest.raises(ValueError):
        supervised.fit([[1, 2]], [[3, 4]], ['a'])
    with pytest.raises(ValueError):
        supervised.fit([[1, float('nan')], [2, 3]], [[3, 4], [4, 5]], ['a', 'b'])
