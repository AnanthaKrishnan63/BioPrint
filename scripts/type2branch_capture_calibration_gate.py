"""Pure completed-study calibration gate; file/hash verification is external."""
import math
import numbers
import re

from type2branch_length_completion import completed_pair


def _number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def calibrated_capture(comparison, arm_reports, calibration_report, calibration_plan):
    """Return four exact-length thresholds only for a completed consistent study."""
    chosen = completed_pair(comparison, arm_reports)
    if not isinstance(calibration_report, dict) or not isinstance(calibration_plan, dict):
        raise ValueError('Calibration report and plan must be objects')
    if calibration_report.get('status') != 'paired_length_train_calibration_complete':
        raise ValueError('Completed length calibration required')
    selected_arm = comparison['selected_arm']
    if (calibration_report.get('selected_arm') != selected_arm
            or calibration_plan.get('selected_arm') != selected_arm
            or calibration_report.get('selected_checkpoint') != chosen
            or calibration_plan.get('checkpoint') != chosen):
        raise ValueError('Calibration is not bound to the selected paired checkpoint')
    lengths = calibration_plan.get('lengths')
    if (not isinstance(lengths, list) or any(type(value) is not int for value in lengths)
            or lengths != [25, 50, 75, 100]):
        raise ValueError('Exactly four ordered integer length anchors required')
    target = calibration_plan.get('target_far')
    if not _number(target) or target != .01:
        raise ValueError('Frozen target FAR must equal0.01')
    results = calibration_report.get('lengths')
    if not isinstance(results, dict) or set(results) != {str(value) for value in lengths}:
        raise ValueError('Exactly four calibration results required')
    thresholds = {}
    for length in lengths:
        result = results[str(length)]
        if not isinstance(result, dict) or not _number(result.get('threshold')):
            raise ValueError('Finite nonboolean threshold required')
        digest = result.get('scores_sha256')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Calibration scores SHA256 required')
        metrics = result.get('metrics')
        pooled = metrics.get('pooled') if isinstance(metrics, dict) else None
        if not isinstance(pooled, dict):
            raise ValueError('Pooled calibration metrics required')
        for name, expected in [('genuine_n', 160), ('impostor_n', 2400)]:
            if type(pooled.get(name)) is not int or pooled[name] != expected:
                raise ValueError('Calibration denominator mismatch')
        for name, maximum in [('false_acceptances', 24), ('false_rejections', 160)]:
            count = pooled.get(name)
            if type(count) is not int or not 0 <= count <= maximum:
                raise ValueError('Invalid calibration error count')
        for name, expected in [('far', pooled['false_acceptances'] / 2400),
                               ('frr', pooled['false_rejections'] / 160)]:
            value = pooled.get(name)
            if (not _number(value) or not 0 <= value <= 1
                    or abs(value - expected) > 1e-12):
                raise ValueError('Calibration rate disagrees with counts')
        eer = pooled.get('discrete_eer')
        if not _number(eer) or not 0 <= eer <= 1:
            raise ValueError('Finite discrete EER within[0,1] required')
        thresholds[length] = float(result['threshold'])
    return thresholds
