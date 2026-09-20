"""Pure equal-weight distance fusion with explicit TRAIN-only scale fitting.

No dataset access, identity joining, threshold selection, or missing-value repair.
Callers must establish identical window/account ordering before constructing the
W x A x M tensor. Modality columns are distances, larger means less similar.
"""
import numpy as np


def _scores(distances, available):
    scores = np.asarray(distances, dtype=np.float64)
    mask = np.asarray(available)
    if (scores.ndim != 3 or any(n == 0 for n in scores.shape)
            or mask.shape != scores.shape or mask.dtype != np.bool_
            or not np.isfinite(scores[mask]).all() or np.any(scores[mask] < 0)):
        raise ValueError('Expected finite nonnegative available W x A x M distances')
    return scores, mask


def fit_scales(distances, available, train_impostor_claims):
    """Mean impostor distance per modality from an explicit TRAIN claim mask.

    Every modality uses the same complete-case claims. Missing or zero scales
    fail the experiment rather than silently replacing or dropping a modality.
    """
    scores, available = _scores(distances, available)
    claims = np.asarray(train_impostor_claims)
    if claims.shape != scores.shape[:2] or claims.dtype != np.bool_:
        raise ValueError('Explicit boolean TRAIN impostor claim mask required')
    selected = claims & available.all(axis=2)
    if not selected.any():
        raise ValueError('No complete TRAIN impostor claims')
    with np.errstate(over='ignore', invalid='ignore'):
        scales = scores[selected].mean(axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Each modality requires a finite positive TRAIN scale')
    return scales


def fuse(distances, available, scales):
    """Equal mean of scaled distances; missing components yield no verdict."""
    scores, available = _scores(distances, available)
    scales = np.asarray(scales, dtype=np.float64)
    if (scales.shape != (scores.shape[2],) or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError('Frozen positive scale for every modality required')
    complete = available.all(axis=2)
    result = np.full(scores.shape[:2], np.nan)
    with np.errstate(over='ignore', invalid='ignore'):
        result[complete] = (scores[complete] / scales).mean(axis=1)
    if not np.isfinite(result[complete]).all():
        raise ValueError('Nonfinite fused score; no verdict')
    return result, complete
