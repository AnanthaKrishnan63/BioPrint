"""Generated/request validation only; no model, artifact, or dataset access."""
import numpy as np


def _reject_non_numeric(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'iuf':
            raise ValueError('Real numeric values required; no booleans or strings')
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_numeric(item)
    elif isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError('Real numeric values required; no booleans or strings')


def validate_features(value):
    """Validate the frozen five-channel contract, returning float32 windows.

    Key codes tolerate only float32 rounding around a byte divided by255.
    Range and first-FT checks precede conversion, preventing rounding from
    concealing invalid input. Signed residuals intentionally remain signed.
    """
    _reject_non_numeric(value)
    try:
        original = np.asarray(value)
        if original.dtype.kind not in 'iuf':
            raise ValueError('Real numeric array required')
        values = np.asarray(original, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Rectangular real numeric windows required') from error
    if values.ndim != 3 or not 1 <= len(values) <= 32 or values.shape[1:] != (100, 5):
        raise ValueError('Expected1..32 windows of shape100x5')
    if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError('Finite float32-representable values required')
    keys = values[:, :, 0]
    if np.any((keys < 0) | (keys > 1)):
        raise ValueError('Normalized key codes must lie in[0,1]')
    codes = np.rint(keys * 255)
    exact = codes / 255
    tolerance = np.spacing(exact.astype(np.float32)).astype(np.float64) / 2
    tolerance += np.finfo(np.float64).eps * np.abs(exact)
    if np.any(np.abs(keys - exact) > tolerance):
        raise ValueError('Normalized key codes must represent integer bytes')
    if np.any((values[:, :, 1:3] < 0) | (values[:, :, 1:3] > 30)):
        raise ValueError('Base HT/FT must lie in[0,30] seconds')
    if np.any(values[:, 0, 2] != 0):
        raise ValueError('First FT must equal zero')
    if np.any((values[:, :, 3:] < -1.5) | (values[:, :, 3:] > 30)):
        raise ValueError('Residuals must lie in[-1.5,30] seconds')
    return values.astype(np.float32)


def validate_request(request):
    """Accept only a correlation ID and windows; no paths or extra options."""
    if not isinstance(request, dict) or set(request) != {'id', 'features'}:
        raise ValueError('Expected exactly id and features')
    identifier = request['id']
    if type(identifier) is not int or identifier < 0:
        raise ValueError('Request id must be a nonnegative integer')
    return identifier, validate_features(request['features'])
