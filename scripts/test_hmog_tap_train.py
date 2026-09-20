import numpy as np
import pytest
from hmog_tap_train import report_scores, split


def metadata():
    counts = [52, 52, 52, 51]
    subjects = np.concatenate([np.repeat(s, n) for s, n in zip(['526319', '539502', '717868', '986737'], counts)])
    session = np.concatenate([np.r_[np.repeat(1, 10), np.repeat(3, 10), np.repeat(9, n - 20)] for n in counts])
    return {'subject': subjects, 'session': session,
            'role': np.where(session <= 8, 'train_enrollment', 'train_fit'),
            'activity_id': np.ones(207, dtype=int), 'window_index': np.arange(207)}


def test_session_split_is_disjoint_and_complete():
    meta = metadata()
    accounts, g, c, d = split(meta)
    assert len(accounts) == 4
    assert (int(g.sum()), int(c.sum()), int(d.sum())) == (40, 40, 127)
    assert np.all(g.astype(int) + c.astype(int) + d.astype(int) == 1)
    assert np.all(meta['session'][g] == 1)
    assert np.all(meta['session'][c] == 3)


def test_test_session_rejected():
    meta = metadata()
    meta['session'][0] = 17
    with pytest.raises(PermissionError):
        split(meta)


def test_diagnostic_does_not_change_thresholds_and_missing_claims_count():
    scores = np.array([[1., 10.], [20., 2.], [3., np.nan], [15., 4.]])
    available = np.isfinite(scores)
    subjects = np.array(['a', 'b', 'a', 'b'])
    calibration = np.array([True, True, False, False])
    diagnostic = ~calibration
    report = report_scores(scores, available, subjects, ['a', 'b'], calibration, diagnostic, [.01])
    assert report['train_diagnostic']['rates']['0.01']['missing_impostor'] == 1
    scores[diagnostic] *= 100
    changed = report_scores(scores, available, subjects, ['a', 'b'], calibration, diagnostic, [.01])
    assert changed['thresholds'] == report['thresholds']
    assert changed['train_diagnostic']['rates']['0.01']['frr'] == 1
