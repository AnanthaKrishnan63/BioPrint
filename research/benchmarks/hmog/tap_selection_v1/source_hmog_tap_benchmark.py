"""Pure tap-reference galleries and window scoring; no IO or population fitting.

All eleven published feature-family dimensions are retained. Numerical choices
and minimum counts are explicit adaptations/contracts, not SOTA claims.
"""
import numpy as np

from hmog_tap_reference import Conventions, fit_profile, scan_mean, scaled_manhattan

CONVENTIONS = Conventions(quantile_method='linear', std_ddof=0, min_spread=0.,
                          velocity_norm='euclidean', zero_spread_policy='ignore')
MIN_ENROLLMENT_TAPS = 80
MIN_WINDOW_TAPS = 5


def validate_taps(taps, tap_window, window_count):
    taps = np.asarray(taps)
    index = np.asarray(tap_window)
    if (type(window_count) is not int or window_count < 1 or taps.ndim != 2
            or taps.shape[1] != 11 or taps.dtype.kind not in 'fi'
            or not np.isfinite(taps).all() or index.shape != (len(taps),)
            or index.dtype.kind not in 'iu' or np.any(index < 0) or np.any(index >= window_count)):
        raise ValueError('Finite tap vectors must map to actual declared windows')
    return taps.astype(np.float64), index


def _window_metadata(subjects, window_mask):
    subjects, mask = np.asarray(subjects), np.asarray(window_mask)
    if (subjects.ndim != 1 or not len(subjects) or subjects.dtype.kind not in 'US'
            or mask.shape != subjects.shape or mask.dtype != np.bool_):
        raise ValueError('String window identities and explicit boolean eligibility required')
    return subjects.astype(str), mask


def build_profiles(taps, tap_window, subjects, enrollment_mask, accounts):
    """Fit per-account enrollment tap statistics, preserving missing galleries.

    The caller enforces cohort/session roles before supplying the enrollment
    mask. Probe rows never contribute merely because their identity matches.
    """
    subjects, enrollment = _window_metadata(subjects, enrollment_mask)
    taps, index = validate_taps(taps, tap_window, len(subjects))
    if (not accounts or accounts != sorted(set(accounts))
            or any(not isinstance(a, str) or not a for a in accounts)
            or not set(subjects).issubset(accounts)):
        raise ValueError('Explicit sorted account population required')
    profiles, diagnostics = {}, {}
    counts = np.bincount(index, minlength=len(subjects))
    for account in accounts:
        # Enrollment only from windows that satisfy the same minimum observed
        # tap support used for authentication scans.
        windows = enrollment & (subjects == account) & (counts >= MIN_WINDOW_TAPS)
        chosen = taps[windows[index]]
        diagnostics[account] = {'enrollment_taps': len(chosen),
                                'enrollment_windows': int(windows.sum())}
        if len(chosen) < MIN_ENROLLMENT_TAPS:
            diagnostics[account]['missing_reason'] = 'fewer_than_80_enrollment_taps'
            continue
        try:
            profiles[account] = fit_profile(chosen, conventions=CONVENTIONS,
                                            min_taps=MIN_ENROLLMENT_TAPS)
        except ValueError as error:
            diagnostics[account]['missing_reason'] = str(error)
    return profiles, diagnostics


def score_windows(taps, tap_window, window_count, profiles, accounts):
    """Scores retain every declared window/account; unavailable claims are NaN.

    Availability is returned explicitly. NaN is never passed to the biometric
    distance function, treated as genuine, or replaced by an invented feature.
    """
    taps, index = validate_taps(taps, tap_window, window_count)
    if not accounts or accounts != sorted(set(accounts)) or not set(profiles).issubset(accounts):
        raise ValueError('Explicit sorted account population required')
    counts = np.bincount(index, minlength=window_count)
    distances = np.full((window_count, len(accounts)), np.nan)
    available = np.zeros_like(distances, dtype=bool)
    for window in np.flatnonzero(counts >= MIN_WINDOW_TAPS):
        scan = scan_mean(taps[index == window])
        for column, account in enumerate(accounts):
            if account in profiles:
                distances[window, column] = scaled_manhattan(profiles[account], scan)
                available[window, column] = True
    return {'distances': distances, 'available': available, 'tap_counts': counts}
