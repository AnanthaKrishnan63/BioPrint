"""Pure frozen-encoder verification; no data loading, fitting, or calibration IO.

Branch outputs are ablations of one jointly trained checkpoint, not independently
optimized unimodal models. Distances are accepted inclusively (distance <= t).
"""
import numpy as np


def _vectors(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 64 or not np.isfinite(result).all():
        raise ValueError('Expected finite N by 64 embeddings')
    return result


def _distances(values, *, nonempty=True):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or (nonempty and not len(result))
            or not np.isfinite(result).all() or np.any(result < 0)):
        raise ValueError('Expected finite nonnegative distance vector')
    return result


def mean_gallery(embeddings, subjects):
    """Arithmetic mean of every supplied eligible enrollment embedding per account.

    Caller enforces enrollment provenance. Missing accounts remain absent.
    """
    vectors = _vectors(embeddings)
    subjects = np.asarray(subjects)
    if subjects.shape != (len(vectors),) or subjects.dtype.kind not in 'US':
        raise ValueError('Expected one string identity per embedding')
    result = {str(s): vectors[subjects == s].mean(axis=0) for s in np.unique(subjects)}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite gallery mean')
    return result


def euclidean_scores(probes, gallery):
    """Return sorted account names and N-probe by M-account distance matrix."""
    probes = _vectors(probes)
    accounts = sorted(gallery)
    if not accounts:
        return accounts, np.empty((len(probes), 0), dtype=np.float64)
    centers = _vectors([gallery[a] for a in accounts])
    # hypot reduction avoids unnecessary overflow from squared large distances.
    with np.errstate(over='ignore', invalid='ignore'):
        scores = np.hypot.reduce(probes[:, None, :] - centers[None, :, :], axis=2)
    if not np.isfinite(scores).all():
        raise ValueError('Unrepresentable Euclidean distance')
    return accounts, scores


def pooled_eer(genuine, impostor):
    """Interpolate FAR/FRR crossing across complete tied-score ROC steps.

    A diagnostic interpolated rate, not an attainable deterministic threshold.
    Both score classes must be present; missing coverage is reported separately.
    """
    g, i = np.sort(_distances(genuine)), np.sort(_distances(impostor))
    thresholds = np.unique(np.concatenate((g, i)))
    frr = np.r_[1., 1 - np.searchsorted(g, thresholds, side='right') / len(g)]
    far = np.r_[0., np.searchsorted(i, thresholds, side='right') / len(i)]
    difference = far - frr
    k = int(np.flatnonzero(difference >= 0)[0])
    if difference[k] == 0:
        return float(far[k])
    weight = -difference[k - 1] / (difference[k] - difference[k - 1])
    return float(far[k - 1] + weight * (far[k] - far[k - 1]))


def far_threshold(impostor, target):
    """Largest float64 acceptance threshold with empirical FAR <= target.

    Ties cannot be partially accepted. No tolerance relaxes the target. At target
    one, return the greatest finite float64; this is not a population guarantee.
    """
    values = np.sort(_distances(impostor))
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('FAR target must lie in [0, 1]')
    unique, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts) / len(values)
    forbidden = np.flatnonzero(cumulative > target)
    if not len(forbidden):
        return float(np.finfo(np.float64).max)
    return float(np.nextafter(unique[forbidden[0]], -np.inf))


def verification_rates(genuine, impostor, threshold, *, expected_genuine=None,
                       expected_impostor=None):
    """Conditional rates and explicit coverage; unavailable claims abstain.

    Expected counts must include missing extraction/gallery claims. Conditional
    FAR/FRR never silently treat abstentions as observed decisions. The separate
    genuine_not_accepted_rate includes abstentions for an end-to-end view.
    """
    g = _distances(genuine, nonempty=False)
    i = _distances(impostor, nonempty=False)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    totals = []
    for expected, observed in ((expected_genuine, len(g)), (expected_impostor, len(i))):
        expected = observed if expected is None else expected
        if (not isinstance(expected, (int, np.integer)) or isinstance(expected, (bool, np.bool_))
                or expected < observed):
            raise ValueError('Expected claim count must be an integer >= observed count')
        totals.append(int(expected))
    ng, ni = totals
    rejected = int(np.count_nonzero(g > threshold))
    accepted = int(np.count_nonzero(i <= threshold))
    return {'frr': rejected / len(g) if len(g) else None,
            'far': accepted / len(i) if len(i) else None,
            'genuine_count': len(g), 'impostor_count': len(i),
            'expected_genuine': ng, 'expected_impostor': ni,
            'missing_genuine': ng - len(g), 'missing_impostor': ni - len(i),
            'genuine_coverage': len(g) / ng if ng else None,
            'impostor_coverage': len(i) / ni if ni else None,
            'genuine_rejections': rejected, 'impostor_acceptances': accepted,
            'genuine_not_accepted_rate': (rejected + ng - len(g)) / ng if ng else None,
            'impostor_acceptances_per_expected_claim': accepted / ni if ni else None}


def branch_embeddings(model, key, imu):
    """Run one eval-only author forward and return joint/key/IMU tensors detached.

    Hooks preserve the exact source execution. No dropout or BatchNorm updates
    are permitted. Inputs must already have their frozen feature scaling.
    """
    import torch
    if any(module.training for module in model.modules()):
        raise ValueError('Encoder and every submodule must be in eval mode')
    if (key.ndim != 3 or key.shape[1:] != (50, 10)
            or imu.shape != (len(key), 100, 24)
            or not torch.isfinite(key).all() or not torch.isfinite(imu).all()):
        raise ValueError('Expected finite paired N×50×10 and N×100×24 tensors')
    captured = {}
    handles = []
    try:
        for name, module in [('key', model.linear_key), ('imu', model.linear_imu),
                             ('joint', model.linear_final)]:
            def capture(_module, _inputs, output, name=name):
                captured[name] = output.detach()
            handles.append(module.register_forward_hook(capture))
        with torch.inference_mode():
            output = model([key, imu])
        if not torch.equal(output, captured['joint']):
            raise ValueError('Final author module differs from model output')
        if any(x.shape != (len(key), 64) or not torch.isfinite(x).all()
               for x in captured.values()):
            raise ValueError('Expected finite 64D branch embeddings')
        return captured
    finally:
        for handle in handles:
            handle.remove()
