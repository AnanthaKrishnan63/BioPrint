"""Pure arithmetic response parsing; no dataset access or fitted parameters.

Missing responses contribute the 2.5-second deadline to bounded waiting time,
not an invented reaction time. Accuracy and missingness remain separate features.
"""
import numpy as np

DEADLINE_SECONDS = 2.5


def scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.size != 1 or arr.dtype.kind not in 'fiu' or not np.isfinite(arr).all():
        raise ValueError('Expected empty or finite numeric scalar')
    return float(arr.reshape(-1)[0])


def parse_block(data):
    # Preserve empty MATLAB cells as slots, including a block of only misses.
    arrays = []
    for name in ['imageT', 'keyT', 'CorrectAns']:
        value = data[name]
        if isinstance(value, (list, tuple)):
            arrays.append(list(value))
        else:
            array = np.asarray(value, dtype=object)
            if array.ndim != 1:
                raise ValueError('Expected one-dimensional MATLAB cell/vector data')
            arrays.append(list(array))
    if len({len(a) for a in arrays}) != 1 or len(arrays[0]) not in (5, 6):
        raise ValueError('Expected five trials with optional trailing padding')
    slots = [tuple(scalar(v) for v in row) for row in zip(*arrays)]
    if len(slots) == 6:
        if slots[-1] != (None, None, 0.0):
            raise ValueError('Sixth slot is not documented TRAIN-observed padding')
        slots = slots[:5]
    onsets, rts, correct = [], [], []
    for onset, press, accuracy in slots:
        if onset is None or accuracy not in (0.0, 1.0):
            raise ValueError('Missing stimulus or invalid correctness')
        if press is None:
            if accuracy != 0:
                raise ValueError('Missing response cannot be correct')
            rt = None
        else:
            rt = press - onset
            if not 0 < rt <= DEADLINE_SECONDS:
                raise ValueError('Response outside task timing contract')
        onsets.append(onset); rts.append(rt); correct.append(int(accuracy))
    if not np.all(np.diff(onsets) > 0):
        raise ValueError('Stimulus order must strictly increase')
    return {'response_time_seconds': rts, 'correct': correct,
            'missing_response': [rt is None for rt in rts],
            'bounded_wait_seconds': [DEADLINE_SECONDS if rt is None else rt for rt in rts]}


def profile(blocks):
    """Combine low/middle/high blocks; these are not simultaneous observations."""
    if set(blocks) != {'l', 'm', 'h'}:
        raise ValueError('All three difficulty levels required')
    waiting, missed, incorrect = [], [], []
    for level in 'lmh':
        block = parse_block(blocks[level])
        waiting.append(float(np.mean(block['bounded_wait_seconds'])))
        missed.append(float(np.mean(block['missing_response'])))
        incorrect.append(float(1 - np.mean(block['correct'])))
    return {'mean_bounded_wait': [float(np.mean(waiting))],
            # OLS slope at equally spaced ordinal labels 0,1,2. This is a
            # contrast, not equal physical difficulty; middle cancels exactly.
            'ordinal_wait_slope': [(waiting[2] - waiting[0]) / 2],
            'condition_bounded_wait': waiting,
            'wait_missing_error_profile': waiting + missed + incorrect}
