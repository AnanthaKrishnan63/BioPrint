"""Explicit residual adaptation, not a recovered paper preprocessing recipe."""
import numpy as np
from type2branch_csv_bridge import integer_rows
from type2branch_synthesis_cleanup import INVALID_TIMING


def residual_features(base, synthesized):
    """Keep normalized base; append signed residual seconds; mask missing to zero."""
    base = np.asarray(base, dtype=np.float64)
    synthesized = integer_rows(synthesized)
    if base.shape != synthesized.shape or not np.isfinite(base).all():
        raise ValueError('Aligned finite N x 3 base required')
    if not np.array_equal(base[:, 0], synthesized[:, 0] / 255.):
        raise ValueError('Normalized key codes must align exactly')
    if (base[:, 1:] < 0).any() or (base[:, 1:] > 30).any():
        raise ValueError('Base timing must use author 0..30s clipping')
    timings = synthesized[:, 1:]
    valid = timings != INVALID_TIMING
    if ((timings[valid] < 0) | (timings[valid] > 1500)).any():
        raise ValueError('Synthetic timings must use source cleanup')
    residual = np.zeros_like(base[:, 1:])
    residual[valid] = base[:, 1:][valid] - timings[valid] / 1000.
    return np.column_stack((base, residual)), valid
