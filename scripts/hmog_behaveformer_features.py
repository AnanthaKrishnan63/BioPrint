"""Pure BehaveFormer feature operators for already guarded/aligned HMOG input.

Source: DilshanSenarath/BehaveFormer commit
319bba27196f18c089841491dc4bf57a1fe90453. Formula/channel order and fixed
divisors follow its HMOG acc+gyro pipeline. Window-local FFT/gradients,
half-open bins, observed lookahead and rejection of missing data are explicit
adaptations; these are not the author's whole-session/overlapping operators.

This module does not read files, select roles, pair events, infer clocks, fit
transforms or instantiate models. Callers must supply verified complete pairs
and same-activity recording-time sensor streams under a frozen data protocol.
Arrays returned are float64; cast only at the inference/training boundary.
"""
from __future__ import annotations
import numpy as np

KEY_CHANNELS = ('hl', 'di_ud', 'di_dd', 'di_uu', 'di_du',
                'tri_ud', 'tri_dd', 'tri_uu', 'tri_du', 'key')
SENSOR_CHANNELS = ('x', 'y', 'z', 'fft_x', 'fft_y', 'fft_z',
                   'fd_x', 'fd_y', 'fd_z', 'sd_x', 'sd_y', 'sd_z')
IMU_CHANNELS = tuple(f'{sensor}_{channel}' for sensor in ('a', 'g') for channel in SENSOR_CHANNELS)


def _array(value, dimensions, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != dimensions or not np.isfinite(result).all():
        raise ValueError(f'{name} must have {dimensions} dimensions and finite values')
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _finite_output(value):
    if not np.isfinite(value).all():
        raise ValueError('Feature computation produced nonfinite values')
    return value


def key_features(press_ms, release_ms, key_ids, *, tokens=50):
    """Return source-scaled (tokens,10) values using tokens+2 actual pairs.

Press order must be nondecreasing; releases may overlap later key presses.
The caller must already have checked contact identity, unique touch anchors
and window membership, including both observed lookahead keys. No pair is
manufactured, no missing feature is padded, and negative special-key IDs stay
numeric. Extra pairs beyond the declared first tokens+2 do not affect output.
"""
    _positive_integer(tokens, 'tokens')
    p = _array(press_ms, 1, 'press_ms')
    r = _array(release_ms, 1, 'release_ms')
    key = _array(key_ids, 1, 'key_ids')
    if not (len(p) == len(r) == len(key)) or len(p) < tokens + 2:
        raise ValueError('Equal-length arrays need at least tokens+2 complete pairs')
    if (p[1:] < p[:-1]).any() or (r < p).any():
        raise ValueError('Presses must be ordered and each release must follow its press')
    if (key != np.trunc(key)).any():
        raise ValueError('Recorded key IDs must be integral')
    p0, r0 = p[:tokens], r[:tokens]
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = np.column_stack((r0-p0, p[1:tokens+1]-r0, p[1:tokens+1]-p0,
                r[1:tokens+1]-r0, r[1:tokens+1]-p0, p[2:tokens+2]-r0,
                p[2:tokens+2]-p0, r[2:tokens+2]-r0, r[2:tokens+2]-p0, key[:tokens]))
            result[:, :9] /= 1000.
            result[:, 9] /= 255.
    except FloatingPointError as error:
        raise ValueError('Keyboard feature overflow') from error
    return _finite_output(result)


def sensor_channels(xyz):
    """Return source unscaled (N,12) channels, requiring N>=3 finite XYZ rows.

FFT is full-length magnitude, not rFFT or spectral power. Derivatives are
per-sample-index np.gradient(edge_order=2), including boundary behavior.
Supply only the window's original records; this function never resamples.
"""
    values = _array(xyz, 2, 'xyz')
    if values.shape[1:] != (3,) or len(values) < 3:
        raise ValueError('XYZ requires at least three rows and exactly three columns')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            spectrum = np.abs(np.fft.fft(values, axis=0))
            first = np.gradient(values, axis=0, edge_order=2)
            second = np.gradient(first, axis=0, edge_order=2)
            result = np.column_stack((values, spectrum, first, second))
    except FloatingPointError as error:
        raise ValueError('IMU feature overflow') from error
    return _finite_output(result)


def _window_bins(times_ms, xyz, start_ms, end_ms, bins):
    times = _array(times_ms, 1, 'recording times')
    values = _array(xyz, 2, 'xyz')
    if values.shape != (len(times), 3) or (times[1:] < times[:-1]).any():
        raise ValueError('Sensor arrays must align and recording times must be nondecreasing')
    inside = (times >= start_ms) & (times < end_ms)
    times, values = times[inside], values[inside]
    # Crucial: discard outside-window context before FFT and differentiation.
    features = sensor_channels(values)
    edges = np.linspace(start_ms, end_ms, bins + 1)
    if not (edges[1:] > edges[:-1]).all():
        raise ValueError('Window/bin width is not representable at supplied timestamp precision')
    assignment = np.searchsorted(edges, times, side='right') - 1
    counts = np.bincount(assignment, minlength=bins)
    if len(counts) != bins or (counts == 0).any():
        raise ValueError('Every half-open bin requires observed sensor support; no padding or interpolation')
    output = np.zeros((bins, len(SENSOR_CHANNELS)), dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise'):
            np.add.at(output, assignment, features)
            output /= counts[:, None]
    except FloatingPointError as error:
        raise ValueError('IMU aggregation overflow') from error
    return _finite_output(output)


def imu_window_features(acc_times_ms, acc_xyz, gyro_times_ms, gyro_xyz, *, start_ms, end_ms, bins=100):
    """Return (bins,24) source-scaled acc+gyro features for [start_ms,end_ms).

Times are caller-verified common *recording* times, not assumed native sensor
or touch clocks. Each stream may have a different sample count; binning pairs
them by the declared interval, never row shape. Equal timestamps keep supplied
row order; no stream is sorted silently. No missing bin or sensor is invented.
"""
    _positive_integer(bins, 'bins')
    if not np.isfinite([start_ms, end_ms]).all() or end_ms <= start_ms:
        raise ValueError('Finite increasing window boundaries required')
    a = _window_bins(acc_times_ms, acc_xyz, start_ms, end_ms, bins)
    g = _window_bins(gyro_times_ms, gyro_xyz, start_ms, end_ms, bins)
    result = np.column_stack((a, g))
    result[:, :3] /= 10.
    result[:, 3:6] /= 1000.
    result[:, 15:18] /= 1000.
    return _finite_output(result)
