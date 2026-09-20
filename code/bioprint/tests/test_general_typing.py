from types import SimpleNamespace
import numpy as np
import pytest
from engine.general_typing import summarize, from_keystrokes, profile, comparison


def test_arbitrary_key_names_and_password_lengths():
    for n in (3, 8, 16, 32):
        strokes = [SimpleNamespace(code=f'Key{i}', down=i * 100, up=i * 100 + 60) for i in range(n)]
        a = from_keystrokes(strokes)
        for key in strokes:
            key.code = 'CompletelyDifferentKey'
        np.testing.assert_array_equal(a, from_keystrokes(strokes))
        assert a.shape == (21,)
        p = profile(np.tile(a, (10, 1)))
        assert comparison(p, [a]).shape == (1, 42)
        assert not comparison(p, [a]).any()


def test_live_and_dataset_channel_parity_with_overlap():
    strokes = [SimpleNamespace(down=0, up=120), SimpleNamespace(down=100, up=180),
               SimpleNamespace(down=200, up=290)]
    np.testing.assert_array_equal(from_keystrokes(strokes), summarize([120, 80, 90], [100, 100], [-20, 20]))


def test_profile_uses_enrollment_only_and_detects_changed_rhythm():
    a = summarize([80, 90, 100], [120, 150], [40, 60])
    p = profile(np.tile(a, (10, 1)))
    center = p[0].copy()
    assert comparison(p, [a * 2])[:, :21].mean() > 0
    np.testing.assert_array_equal(p[0], center)
    assert np.isfinite(comparison(p, [a])).all()


@pytest.mark.parametrize('hold', [[-1, 2], [float('nan'), 2], [1]])
def test_invalid_timings_fail_closed(hold):
    with pytest.raises(ValueError):
        summarize(hold, [1, 2], [1, 2])


def test_insufficient_enrollment_rejected():
    with pytest.raises(ValueError):
        profile(np.zeros((9, 21)))
