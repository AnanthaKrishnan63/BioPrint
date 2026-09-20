"""Pure event adaptation for frozen BEACON windows; no data loading or fitting.

Sorting and short (5..24 event) inputs are explicit transfer adaptations. The
caller owns window eligibility, population provenance, and RNG stream lifetime.
"""
from decimal import Decimal, InvalidOperation
import math

import numpy as np
from type2branch_variable_adapter import synthesize_window


def milliseconds(text):
    try:
        value = Decimal(text) * 1000
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError('Invalid elapsed time') from error
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError('Elapsed timestamp is not exact integer milliseconds')
    return int(value)


def parse_rows(rows, *, key_mapper=None):
    """Return onset-sorted integer-ms events and recording-level diagnostics.

Duration is diagnostic only: observed hold is elapsed release minus press.
Inject a mapper for isolated generated-data checks; production uses the existing
BEACON mapping, including its documented unknown-code zero convention.
"""
    if key_mapper is None:
        from beacon_neural_benchmark import mapped_key
        key_mapper = mapped_key
    events = []
    disagreement = []
    for row in rows:
        press = milliseconds(row['Elapsed Start Time'])
        release = milliseconds(row['Elapsed Release Time'])
        duration = milliseconds(row['Duration'])
        if release < press:
            raise ValueError('Negative hold')
        code = key_mapper(row['Key'])
        if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or not 0 <= code <= 255:
            raise ValueError('Invalid mapped key code')
        events.append((press, release, int(code)))
        disagreement.append(abs(release - press - duration))
    if not events:
        raise ValueError('No observed keys')
    events = np.asarray(events, dtype=np.int64)
    decreases = int(np.count_nonzero(np.diff(events[:, 0]) < 0))
    events = events[np.argsort(events[:, 0], kind='stable')]
    return {'events_ms': events, 'audit': {
        'events': len(events),
        'unknown_codes': int(np.count_nonzero(events[:, 2] == 0)),
        'release_order_onset_decreases': decreases,
        'median_duration_disagreement_ms': float(np.median(disagreement)),
        'maximum_duration_disagreement_ms': max(disagreement),
    }}


def adapt_window(parsed, start):
    """Keep complete holds in [start, start+30), then the first 100 events.

Float-second comparisons deliberately match the prior paired-window pipeline.
Never round the supplied window start or infer a minimum 25-event anchor.
"""
    if isinstance(start, bool) or not isinstance(start, (int, float, np.number)) or not math.isfinite(start):
        raise ValueError('Finite numeric window start required')
    events = np.asarray(parsed['events_ms'])
    if events.ndim != 2 or events.shape[1] != 3 or events.dtype.kind not in 'iu':
        raise ValueError('Expected integer-ms events')
    if (events[:, 1] < events[:, 0]).any() or (np.diff(events[:, 0]) < 0).any():
        raise ValueError('Expected nonnegative holds sorted by onset')
    times = events[:, :2] / 1000.
    selected = events[(times[:, 0] >= start) & (times[:, 1] < start + 30)]
    count = len(selected)
    if count < 5:
        raise ValueError('Old accepted window no longer meets baseline key minimum')
    selected = selected[:100]
    length = len(selected)
    wire = np.column_stack((selected[:, 2], selected[:, 1] - selected[:, 0],
                            np.r_[0, np.diff(selected[:, 0])])).astype(np.int64)
    base = wire.astype(np.float64)
    base[:, 0] /= 255.
    base[:, 1:] = np.clip(base[:, 1:] / 1000., 0, 30)
    return {'true_length': length, 'raw_ms': wire, 'base': base}, {
        'start': float(start), 'eligible_events': count, 'true_length': length,
        'truncated_events': count - length, 'below_source_minimum_25': length < 25,
        'unknown_codes': int(np.count_nonzero(wire[:, 0] == 0)),
    }


def synthesize_rows(rows, start, population, rng, *, key_mapper=None):
    """Convenience adapter; synthesis consumes only real events before padding."""
    parsed = parse_rows(rows, key_mapper=key_mapper)
    window, audit = adapt_window(parsed, start)
    return synthesize_window(window, population, rng), {
        'recording': parsed['audit'], 'window': audit,
    }
