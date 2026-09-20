"""Fixed-threshold DEV reporting without ranking, calibration, or promotion."""
import numpy as np
from hmog_verification import pooled_eer, verification_rates

METHODS = ('tap11', 'key', 'imu', 'joint', 'key_imu_tap11', 'joint_tap11')


def evaluate(scores, available, subjects, accounts, probe_mask, thresholds,
             expected_candidate_probes):
    scores = np.asarray(scores, dtype=np.float64)
    available = np.asarray(available)
    subjects = np.asarray(subjects).astype(str)
    probe = np.asarray(probe_mask)
    if (subjects.ndim != 1 or len(accounts) != 4 or accounts != sorted(set(accounts))
            or not set(subjects).issubset(accounts) or scores.shape != (len(subjects), 4)
            or available.shape != scores.shape or available.dtype != np.bool_
            or probe.shape != subjects.shape or probe.dtype != np.bool_
            or not np.isfinite(scores[available]).all() or np.any(scores[available] < 0)
            or set(thresholds) != {'0.001', '0.01', '0.05'}
            or not all(np.isfinite(v) for v in thresholds.values())
            or type(expected_candidate_probes) is not int
            or expected_candidate_probes < int(probe.sum())):
        raise ValueError('Explicit four-account DEV scores, masks and coverage required')
    genuine = subjects[:, None] == np.asarray(accounts)[None, :]
    observed = probe[:, None] & available
    g, i = scores[observed & genuine], scores[observed & ~genuine]
    base_count = int(probe.sum())

    def rates(expected):
        return {t: verification_rates(g, i, value, expected_genuine=expected,
                                       expected_impostor=expected * 3)
                for t, value in thresholds.items()}

    per_account = {}
    for column, account in enumerate(accounts):
        rows = probe & (subjects == account)
        actual_g = scores[rows & available[:, column], column]
        actual_i = scores[probe & (subjects != account) & available[:, column], column]
        per_account[account] = {
            'original_genuine_probes': int(rows.sum()),
            'missing_genuine_probe_account': not rows.any(),
            'eer': pooled_eer(actual_g, actual_i) if len(actual_g) and len(actual_i) else None,
            'rates': {t: verification_rates(actual_g, actual_i, value,
                expected_genuine=int(rows.sum()), expected_impostor=int((probe & ~rows).sum()))
                for t, value in thresholds.items()}}
    return {'eer': pooled_eer(g, i) if len(g) and len(i) else None,
            'base_window_rates': rates(base_count),
            'known_inspected_candidate_rates': rates(expected_candidate_probes),
            'per_account': per_account,
            'unknown_excluded_session_candidate_counts_remain_unknown': True}


def summarize(scores, available, subjects, accounts, probe_mask, thresholds,
              expected_candidate_probes=154):
    if any(set(d) != set(METHODS) for d in (scores, available, thresholds)):
        raise ValueError('All six fixed comparison methods required')
    base = {m: evaluate(scores[m], available[m], subjects, accounts, probe_mask,
                         thresholds[m], expected_candidate_probes) for m in METHODS}
    intersection = np.logical_and.reduce([available[m] for m in METHODS])
    shared = {m: evaluate(scores[m], intersection, subjects, accounts, probe_mask,
                           thresholds[m], expected_candidate_probes) for m in METHODS}
    subjects, probe = np.asarray(subjects).astype(str), np.asarray(probe_mask)
    missing = [a for a in accounts if not np.any(probe & (subjects == a))]
    return {'status': 'infeasible' if missing else 'complete_evaluation_only',
            'missing_genuine_probe_accounts': missing,
            'observed_window_comparison_completed': True,
            'metrics': {m: {'base_coverage': base[m], 'shared_intersection': shared[m]} for m in METHODS},
            'selection_outcome': 'no_useful_discrimination',
            'no_method_ranking_or_promotion': True,
            'original_windows': len(subjects), 'original_probe_windows': int(probe.sum()),
            'known_inspected_candidate_probes': expected_candidate_probes}
