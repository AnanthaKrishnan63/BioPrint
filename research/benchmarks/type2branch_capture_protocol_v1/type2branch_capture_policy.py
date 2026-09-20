"""Fixed first-capture allocation policy; no data access or fitted parameters."""
from type2branch_variable_adapter import adapt_variable

ANCHORS = (25, 50, 75, 100)


def allocate_capture(rows):
    """Use the largest calibrated prefix within the first100 available rows.

    Short captures require additional verification without timing parsing.
    Adapter errors propagate: required malformed rows are never skipped.
    """
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise ValueError('Expected a list of opaque source-row strings')
    available = len(rows)
    if available < ANCHORS[0]:
        return None, {'available_events': available, 'parsed_events': 0,
                      'used_length': 0, 'excluded_events': available,
                      'action': 'additional_verification',
                      'reason': 'Fewer than25 events; no calibrated capture length'}
    length = max(anchor for anchor in ANCHORS if anchor <= available)
    windows, details = adapt_variable(rows[:length], max_windows=1,
                                      session_complete=(available == length))
    if (len(windows) != 1 or windows[0]['true_length'] != length
            or details['parsed_events'] != length):
        raise ValueError('Adapter did not preserve the exact capture allocation')
    return windows[0], {'available_events': available, 'parsed_events': length,
                       'used_length': length, 'excluded_events': available - length,
                       'action': 'score',
                       'reason': 'Largest calibrated anchor within first100 events',
                       'adapter': details}
