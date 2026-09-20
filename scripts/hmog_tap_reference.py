"""Documented HMOG tap-11 family with explicit, non-reproduction conventions.

Input: already validated complete single-pointer contacts, event milliseconds,
pixel x/y, raw contact size. No event repair, key content, sensors, or dataset IO.
"""
from dataclasses import asdict, dataclass
import json
import numpy as np

FEATURE_NAMES = ('duration_ms', 'size_mean', 'size_median', 'size_std', 'size_q1',
                 'size_q2', 'size_q3', 'size_first', 'size_min', 'size_max',
                 'press_velocity_pixels_per_second')
SCHEMA = 'hmog_tap11_explicit_conventions_v1'


@dataclass(frozen=True)
class Conventions:
    """Required caller choices where the paper does not fully specify numerics."""
    quantile_method: str
    std_ddof: int
    min_spread: float
    velocity_norm: str
    zero_spread_policy: str

    def validate(self):
        if self.quantile_method != 'linear' or self.velocity_norm != 'euclidean':
            raise ValueError('This adaptation implements linear quantiles and Euclidean velocity only')
        if type(self.std_ddof) is not int or self.std_ddof not in (0, 1):
            raise ValueError('Explicit std_ddof must be 0 or 1')
        if (isinstance(self.min_spread, bool) or not isinstance(self.min_spread, (int, float))
                or not np.isfinite(self.min_spread) or self.min_spread < 0 or self.zero_spread_policy != 'ignore'):
            raise ValueError('Specify finite nonnegative min_spread and ignore policy')


@dataclass(frozen=True)
class Contact:
    """Includes physical DOWN first and UP last; upstream owns event pairing."""
    session: str
    event_ms: tuple
    x_pixels: tuple
    y_pixels: tuple
    sizes: tuple

    def array(self):
        try:
            a = np.asarray([self.event_ms, self.x_pixels, self.y_pixels, self.sizes], dtype=np.float64).T
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Invalid contact arrays') from exc
        if a.ndim != 2 or a.shape[1] != 4 or a.shape[0] < 2 or not np.isfinite(a).all():
            raise ValueError('Contact needs at least DOWN/UP finite samples')
        with np.errstate(over='ignore', invalid='ignore'):
            delta = np.diff(a[:, 0])
        if not np.isfinite(delta).all() or np.any(delta <= 0) or np.any(a[:, 3] < 0):
            raise ValueError('Reject duplicate/reversed clocks and negative contact sizes')
        if not isinstance(self.session, str) or not self.session:
            raise ValueError('Explicit session identity required')
        return a


def tap11(contact, previous_contact, *, conventions):
    """Velocity uses immediately preceding complete contact; no first-tap imputation.

    Caller guarantees no intervening invalid/missing contact and same coordinate
    frame. First tap after session/reset/gap is unavailable, not a zero velocity.
    """
    conventions.validate()
    if previous_contact is None:
        raise ValueError('Previous complete consecutive contact required')
    current, previous = contact.array(), previous_contact.array()
    if contact.session != previous_contact.session or previous[-1, 0] > current[0, 0]:
        raise ValueError('Do not bridge sessions or overlapping contacts')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        gap = current[0, 0] - previous[0, 0]
        velocity = np.hypot(*(current[0, 1:3] - previous[0, 1:3])) / gap * 1000
        sizes = current[:, 3]
        q1, q2, q3 = np.quantile(sizes, [.25, .5, .75], method=conventions.quantile_method)
        v = np.asarray([current[-1, 0] - current[0, 0], np.mean(sizes), np.median(sizes),
                        np.std(sizes, ddof=conventions.std_ddof), q1, q2, q3,
                        sizes[0], np.min(sizes), np.max(sizes), velocity])
    if not np.isfinite(v).all() or gap <= 0:
        raise ValueError('Nonfinite/invalid feature arithmetic')
    return v


def matrix(rows):
    try:
        a = np.asarray(rows, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid feature matrix') from exc
    if a.ndim != 2 or a.shape[1] != len(FEATURE_NAMES) or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite N x 11 matrix')
    return a


def scan_mean(tap_vectors):
    """Unweighted mean; caller already enforces shared-session/window boundaries."""
    with np.errstate(over='ignore', invalid='ignore'):
        result = matrix(tap_vectors).mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite scan mean')
    return result


def fit_profile(enrollment_taps, *, conventions, min_taps):
    """Enrollment tap-wise mean/std; no population data or probe adaptation."""
    conventions.validate()
    a = matrix(enrollment_taps)
    if type(min_taps) is not int or min_taps < 2 or len(a) < min_taps:
        raise ValueError('Insufficient enrollment; explicit min_taps >=2 required')
    with np.errstate(over='ignore', invalid='ignore'):
        mean, spread = a.mean(axis=0), a.std(axis=0, ddof=conventions.std_ddof)
    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
        raise ValueError('Nonfinite enrollment arithmetic')
    active = spread > conventions.min_spread
    if not active.any():
        raise ValueError('No enrollment feature has usable spread; no verdict')
    return validate_profile({'schema': SCHEMA, 'features': list(FEATURE_NAMES), 'conventions': asdict(conventions),
                             'n_enrollment': len(a), 'min_taps': min_taps, 'mean': mean.tolist(),
                             'spread': spread.tolist(), 'active': active.tolist(),
                             'active_feature_count': int(active.sum()),
                             'score_direction': 'larger_is_impostor', 'distance_reduction': 'sum'})


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get('schema') != SCHEMA or profile.get('features') != list(FEATURE_NAMES):
        raise ValueError('Unknown profile schema/features')
    try:
        config = Conventions(**profile['conventions']); config.validate()
        mean = matrix([profile['mean']])[0]; spread = matrix([profile['spread']])[0]
        active = profile['active']
        if (len(active) != 11 or any(type(x) is not bool for x in active) or
                np.any(spread < 0) or active != (spread > config.min_spread).tolist() or not any(active)
                or type(profile['active_feature_count']) is not int or profile['active_feature_count'] != sum(active)):
            raise ValueError('Invalid spread/active feature policy')
        if (type(profile['n_enrollment']) is not int or type(profile['min_taps']) is not int
                or profile['min_taps'] < 2 or profile['n_enrollment'] < profile['min_taps']
                or profile['score_direction'] != 'larger_is_impostor' or profile['distance_reduction'] != 'sum'):
            raise ValueError('Invalid enrollment or score convention')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid profile fields') from exc
    return profile


def scaled_manhattan(profile, authentication_vector):
    validate_profile(profile)
    vector = matrix([authentication_vector])[0]
    active = np.asarray(profile['active'], dtype=bool)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = np.sum(np.abs(vector[active] - np.asarray(profile['mean'])[active]) / np.asarray(profile['spread'])[active])
    if not np.isfinite(result):
        raise ValueError('Nonfinite distance; no verdict')
    return float(result)


def profile_json(profile):
    return json.dumps(validate_profile(profile), allow_nan=False, sort_keys=True)


def profile_from_json(payload):
    try:
        value = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON literal')))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid profile JSON') from exc
    return validate_profile(value)
