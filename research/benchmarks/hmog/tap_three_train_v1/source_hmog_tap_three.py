"""Pure fixed published tap subset; no IO, fitting, or feature selection.

Uses duration, mean contact size, and inter-press velocity (full11 indices0,1,10).
This is a source-subset adaptation, not a reproduction of mRMR selection. Gallery
means/spreads must come from the same authorized enrollment taps as full11.
"""
import numpy as np

from hmog_tap_benchmark import MIN_ENROLLMENT_TAPS, MIN_WINDOW_TAPS, validate_taps
from hmog_tap_reference import scan_mean, validate_profile

FEATURE_INDICES = (0, 1, 10)
FEATURE_NAMES = ('duration_ms', 'size_mean', 'press_velocity_pixels_per_second')


def _active(profile):
    validate_profile(profile)
    if profile['n_enrollment'] < MIN_ENROLLMENT_TAPS:
        raise ValueError('Fixed3 requires the same minimum80 enrollment taps')
    return np.asarray(FEATURE_INDICES)[np.asarray(profile['active'], dtype=bool)[list(FEATURE_INDICES)]]


def scaled_manhattan_three(profile, authentication_vector):
    """Selected-dimension sum, or None if all three have unusable spread.

    Zero-spread dimensions are ignored according to the stored full11 profile's
    convention. Never divide by the number of selected or active dimensions.
    """
    active = _active(profile)
    vector = np.asarray(authentication_vector, dtype=np.float64)
    if vector.shape != (11,) or not np.isfinite(vector).all():
        raise ValueError('Expected finite full11 authentication vector')
    if not len(active):
        return None
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        distance = np.sum(np.abs(vector[active] - np.asarray(profile['mean'])[active]) /
                          np.asarray(profile['spread'])[active])
    if not np.isfinite(distance):
        raise ValueError('Nonfinite fixed3 distance; no verdict')
    return float(distance)


def score_windows(taps, tap_window, window_count, profiles, accounts):
    """Preserve every original window/account slot with an explicit validity mask."""
    taps, index = validate_taps(taps, tap_window, window_count)
    if (not accounts or accounts != sorted(set(accounts))
            or any(not isinstance(a, str) or not a for a in accounts)
            or not set(profiles).issubset(accounts)):
        raise ValueError('Explicit sorted account population required')
    active_counts = {a: len(_active(profiles[a])) if a in profiles else 0 for a in accounts}
    counts = np.bincount(index, minlength=window_count)
    distances = np.full((window_count, len(accounts)), np.nan)
    available = np.zeros_like(distances, dtype=bool)
    for window in np.flatnonzero(counts >= MIN_WINDOW_TAPS):
        vector = scan_mean(taps[index == window])
        for column, account in enumerate(accounts):
            if active_counts[account]:
                distances[window, column] = scaled_manhattan_three(profiles[account], vector)
                available[window, column] = True
    return {'distances': distances, 'available': available, 'tap_counts': counts,
            'active_feature_counts': active_counts}
