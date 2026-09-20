"""Pure frozen-threshold TRAIN selection ranking with abstention accounting."""
import numpy as np
from hmog_verification import pooled_eer, verification_rates

METHOD_ORDER = ('joint_tap11', 'tap11', 'key', 'imu', 'joint', 'key_imu_tap11')


def evaluate(scores, available, subjects, accounts, probe_mask, thresholds):
    """Summarize one method, preserving the original per-person denominator.

    Caller verifies cohort/role/window provenance and frozen threshold hashes.
    This function fits nothing and never substitutes an EER-derived threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    available = np.asarray(available)
    subjects = np.asarray(subjects).astype(str)
    probe = np.asarray(probe_mask)
    if (subjects.ndim != 1 or len(accounts) != 2 or accounts != sorted(set(accounts))
            or not set(subjects).issubset(accounts)
            or scores.shape != (len(subjects), len(accounts))
            or available.shape != scores.shape or available.dtype != np.bool_
            or probe.shape != subjects.shape or probe.dtype != np.bool_
            or not np.isfinite(scores[available]).all() or np.any(scores[available] < 0)
            or set(thresholds) != {'0.001', '0.01', '0.05'}
            or not all(np.isfinite(v) for v in thresholds.values())):
        raise ValueError('Explicit two-account scores, masks, and frozen thresholds required')
    genuine = subjects[:, None] == np.asarray(accounts)[None, :]
    observed = probe[:, None] & available
    g, i = scores[observed & genuine], scores[observed & ~genuine]
    expected = int(probe.sum())
    rates = {target: verification_rates(g, i, value, expected_genuine=expected,
                                        expected_impostor=expected)
             for target, value in thresholds.items()}
    per_account = {}
    for column, account in enumerate(accounts):
        rows = probe & (subjects == account)
        n = int(rows.sum())
        if n == 0:
            raise ValueError('Every original account requires a probe denominator')
        accept = rows & available[:, column] & (scores[:, column] <= thresholds['0.01'])
        accepted = int(accept.sum())
        per_account[account] = {'expected_genuine': n, 'accepted_genuine': accepted,
                                'genuine_not_accepted_rate': (n - accepted) / n}
    return {'eer': pooled_eer(g, i) if len(g) and len(i) else None,
            'rates': rates, 'per_account_at_1pct': per_account,
            'macro_genuine_not_accepted': float(np.mean([
                a['genuine_not_accepted_rate'] for a in per_account.values()])),
            'qualifies': bool(len(i) and rates['0.01']['impostor_acceptances'] == 0),
            'accepted_genuine': sum(a['accepted_genuine'] for a in per_account.values()),
            'population_low_far_proven': False}


def rank(reports):
    """Fixed macro rejection objective, zero-observed-FA gate, declared ties."""
    if set(reports) != set(METHOD_ORDER):
        raise ValueError('All six prespecified methods must be reported')
    candidates = [name for name in METHOD_ORDER if reports[name]['qualifies']]
    if not candidates:
        return {'status': 'infeasible', 'selected_method': None,
                'reason': 'no_method_passes_observed_false_accept_gate'}
    chosen = min(candidates, key=lambda name: reports[name]['macro_genuine_not_accepted'])
    useful = reports[chosen]['accepted_genuine'] > 0
    return {'status': 'selected' if useful else 'no_useful_discrimination',
            'selected_method': chosen,
            'macro_genuine_not_accepted': reports[chosen]['macro_genuine_not_accepted'],
            'qualifying_methods': candidates, 'population_low_far_proven': False}


def summarize(scores, available, subjects, accounts, probe_mask, thresholds):
    """Report each method on its base coverage and the common claim intersection."""
    if any(set(d) != set(METHOD_ORDER) for d in (scores, available, thresholds)):
        raise ValueError('All six prespecified method inputs required')
    base = {name: evaluate(scores[name], available[name], subjects, accounts,
                           probe_mask, thresholds[name]) for name in METHOD_ORDER}
    intersection = np.logical_and.reduce([available[name] for name in METHOD_ORDER])
    shared = {name: evaluate(scores[name], intersection, subjects, accounts,
                             probe_mask, thresholds[name]) for name in METHOD_ORDER}
    selection = rank(shared)
    return {**selection, 'metrics': {
        name: {'base_coverage': base[name], 'shared_intersection': shared[name]}
        for name in METHOD_ORDER},
        'selection_scope': 'Shared original claim intersection; original per-account probe denominators',
        'window_count': len(subjects), 'probe_windows': int(np.asarray(probe_mask).sum())}
