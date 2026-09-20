"""Conditional recognition metrics and explicit whole-cohort capture coverage."""
import numbers

import numpy as np

from keystroke_benchmark import curve_metrics

ANCHORS = (25, 50, 75, 100)


def _finite_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def capture_metrics(scores, probe_subjects, lengths, identities, coverage, thresholds, cohorts):
    """Use exact-length thresholds, retaining absent captures outside score rates.

    Eligible score rejections and unavailable captures both request additional
    verification, but are separately counted. Cohort subsets keep the original
    all-account claims unless explicitly named within_cohort.
    """
    identities = list(identities)
    if (len(identities) < 2 or not all(isinstance(s, str) and s for s in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError('Distinct string account identities required')
    universe = set(identities)
    if not isinstance(coverage, dict) or set(coverage) != universe:
        raise ValueError('Coverage must describe every account exactly')
    if (not isinstance(thresholds, dict) or set(thresholds) != set(ANCHORS)
            or any(type(k) is not int for k in thresholds)
            or not all(_finite_number(v) for v in thresholds.values())):
        raise ValueError('Four exact integer anchors with finite thresholds required')
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError('Named exhaustive cohorts required')
    grouped = {}
    seen = set()
    for name, members in cohorts.items():
        if (not isinstance(name, str) or not name or not isinstance(members, (list, tuple))
                or not all(isinstance(s, str) for s in members)
                or len(set(members)) != len(members) or seen.intersection(members)
                or not set(members).issubset(universe)):
            raise ValueError('Cohorts must be disjoint account lists')
        grouped[name] = set(members)
        seen.update(members)
    if seen != universe:
        raise ValueError('Cohorts must exhaust accounts')
    eligible = set()
    ineligible = []
    for identity in identities:
        item = coverage[identity]
        if not isinstance(item, dict):
            raise ValueError('Coverage records must be objects')
        action, length = item.get('action'), item.get('used_length')
        if type(length) is not int:
            raise ValueError('Coverage lengths must be integers')
        if action == 'score' and length in ANCHORS:
            eligible.add(identity)
        elif action == 'additional_verification' and length == 0:
            reason = item.get('reason')
            if not isinstance(reason, str) or not reason:
                raise ValueError('Unavailable captures require a reason')
            ineligible.append({'identity': identity, 'reason': reason})
        else:
            raise ValueError('Invalid coverage action or length')
    subjects = list(probe_subjects)
    if (not all(isinstance(s, str) for s in subjects) or len(set(subjects)) != len(subjects)
            or set(subjects) != eligible):
        raise ValueError('Exactly one probe per eligible identity required')
    sizes = np.asarray(lengths)
    if (sizes.shape != (len(subjects),)
            or (len(subjects) and sizes.dtype.kind not in 'iu')
            or any(isinstance(v, (bool, np.bool_)) for v in lengths)):
        raise ValueError('One integer capture length required per probe')
    if any(int(length) != coverage[s]['used_length'] for s, length in zip(subjects, sizes)):
        raise ValueError('Probe length differs from coverage allocation')
    values = np.asarray(scores)
    if (values.dtype.kind not in 'iuf' or values.shape != (len(subjects), len(identities))):
        raise ValueError('Expected finite real Q x U scores')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite scores')
    row_thresholds = np.array([thresholds[int(length)] for length in sizes], dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        margins = values - row_thresholds[:, None]
    if not np.isfinite(margins).all():
        raise ValueError('Threshold subtraction overflow')
    accepted = values >= row_thresholds[:, None]
    labels = np.array([identities.index(s) for s in subjects], dtype=int)

    def subset(row_indices, claim_indices):
        if not len(row_indices) or len(claim_indices) < 2:
            return {}
        claims = np.asarray(claim_indices)
        is_genuine = labels[row_indices, None] == claims[None, :]
        if not is_genuine.any() or not (~is_genuine).any():
            return {}
        raw = values[np.ix_(row_indices, claims)]
        adjusted = margins[np.ix_(row_indices, claims)]
        decisions = accepted[np.ix_(row_indices, claims)]
        genuine_n, impostor_n = int(is_genuine.sum()), int((~is_genuine).sum())
        rejects = int((~decisions[is_genuine]).sum())
        accepts = int(decisions[~is_genuine].sum())
        return {'genuine_n': genuine_n, 'impostor_n': impostor_n,
                'false_rejections': rejects, 'false_acceptances': accepts,
                'frr': rejects / genuine_n, 'far': accepts / impostor_n,
                'discrete_eer': curve_metrics(raw[is_genuine], raw[~is_genuine])['eer'],
                'margin_discrete_eer': curve_metrics(adjusted[is_genuine], adjusted[~is_genuine])['eer']}

    total, count = len(identities), len(subjects)
    conditional = {}
    if count:
        rows, claims = np.arange(count), np.arange(total)
        cohort_rows = {name: np.array([i for i, s in enumerate(subjects) if s in members], dtype=int)
                       for name, members in grouped.items()}
        conditional = {'pooled': subset(rows, claims),
            'per_length': {str(length): subset(np.flatnonzero(sizes == length), claims) for length in ANCHORS},
            'probe_cohorts_full_claims': {name: subset(indices, claims) for name, indices in cohort_rows.items()},
            'within_cohort': {name: subset(cohort_rows[name],
                np.array([i for i, s in enumerate(identities) if s in members], dtype=int))
                for name, members in grouped.items()}}
    genuine_accepts = int(accepted[np.arange(count), labels].sum())
    genuine_rejections = count - genuine_accepts
    cohort_coverage = {}
    for name, members in grouped.items():
        available = len(eligible.intersection(members))
        cohort_coverage[name] = {'eligible': available, 'total': len(members),
            'fraction': available / len(members) if members else None,
            'ineligible': [item for item in ineligible if item['identity'] in members]}
    length_coverage = {str(length): {'eligible': int(np.sum(sizes == length)),
        'total': total, 'fraction_of_all_accounts': int(np.sum(sizes == length)) / total}
        for length in ANCHORS}
    return {'coverage': {'eligible': count, 'total': total, 'fraction': count / total,
                         'ineligible': ineligible, 'ineligible_count': total - count,
                         'by_cohort': cohort_coverage, 'by_length': length_coverage},
            'conditional_metrics': conditional,
            'whole_cohort': {'total': total, 'direct_acceptances': genuine_accepts,
                'direct_acceptance_fraction': genuine_accepts / total,
                'score_rejections': genuine_rejections,
                'score_rejection_fraction': genuine_rejections / total,
                'ineligible': total - count, 'ineligible_fraction': (total - count) / total,
                'additional_verifications': total - genuine_accepts,
                'additional_verification_fraction': (total - genuine_accepts) / total},
            'interpretation': 'Conditional rates exclude unavailable captures; raw-score and threshold-margin EER are diagnostic'}
