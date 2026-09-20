"""Pure completion gate; callers separately verify source/artifact hashes."""
import math
import numbers
import re

from type2branch_length_training_core import selection_key

ARMS = ('full', 'mixed')


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Expected a SHA256 batch-stream digest')
    return value


def completed_pair(pair_report, arm_reports):
    """Require both equal-budget arms and the frozen deterministic selection.

    Exact ties between arms prefer full. This checks report consistency, not
    the contents or provenance of checkpoints, arrays, or source files.
    """
    if not isinstance(pair_report, dict) or not isinstance(arm_reports, dict):
        raise ValueError('Pair and arm reports must be objects')
    if set(arm_reports) != set(ARMS):
        raise ValueError('Exactly full and mixed arm reports required')
    best = {}
    streams = {}
    for arm in ARMS:
        report = arm_reports[arm]
        if (not isinstance(report, dict) or report.get('status') != 'paired_arm_complete'
                or report.get('arm') != arm or type(report.get('updates')) is not int
                or report['updates'] != 600):
            raise ValueError('Both named arms must complete600 optimizer updates')
        records = report.get('checkpoints')
        if not isinstance(records, list) or len(records) != 4:
            raise ValueError('Four ordered checkpoints required per arm')
        for epoch, record in zip(range(2, 6), records):
            if (not isinstance(record, dict) or type(record.get('epoch')) is not int
                    or record['epoch'] != epoch or type(record.get('updates')) is not int
                    or record['updates'] != (epoch + 1) * 100
                    or not isinstance(record.get('checkpoint'), str)
                    or not record['checkpoint']):
                raise ValueError('Checkpoint epoch, update count or path mismatch')
            selection_key(record)
        chosen = min(records, key=selection_key)
        if report.get('best_checkpoint') != chosen:
            raise ValueError('Arm best checkpoint violates frozen selection rule')
        best[arm] = chosen
        streams[arm] = _hash(report.get('batch_stream_sha256'))
    if streams['full'] != streams['mixed']:
        raise ValueError('Paired input batch streams differ')
    selected_arm = min(ARMS, key=lambda arm: selection_key(best[arm]))
    if (pair_report.get('status') != 'paired_train_comparison_complete'
            or pair_report.get('selected_arm') != selected_arm
            or pair_report.get('arm_best') != best
            or pair_report.get('selected_checkpoint') != best[selected_arm]
            or _hash(pair_report.get('paired_batch_stream_sha256')) != streams['full']):
        raise ValueError('Pair report disagrees with completed arms')
    error = pair_report.get('max_warm_start_score_error')
    if (isinstance(error, bool) or not isinstance(error, numbers.Real)
            or not math.isfinite(error) or not 0 <= error <= 1e-6):
        raise ValueError('Warm-start error must be finite and within[0,1e-6]')
    return best[selected_arm]
