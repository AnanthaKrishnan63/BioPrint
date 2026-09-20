"""Strict wire-format primitives; not a reproduction of synthesis/preprocessing.

Rows are unpadded [VK, HT_ms, FT_ms]. No rounding, clipping, sentinel cleanup,
population fitting, or dataset access occurs here. Timing cleanup and the
paper/code clipping discrepancy must be resolved before model integration.
"""
import csv
import io
import re

import numpy as np


def integer_rows(rows):
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError('Expected a nonempty unpadded N x 3 array')
    if not np.isfinite(values).all() or not np.equal(values, np.trunc(values)).all():
        raise ValueError('Integer milliseconds and key codes required; no implicit rounding')
    if np.any(values[:, 0] < 0) or np.any(values[:, 0] > 255):
        raise ValueError('Key codes must fit the C# byte domain')
    if np.any(values[:, 1:] < -(2**31)) or np.any(values[:, 1:] > 2**31 - 1):
        raise ValueError('Timing values must fit C# Int32')
    return values.astype(np.int64)


def encode_csv(rows):
    """Encode actual events only; callers must retain the true sequence length."""
    rows = integer_rows(rows)
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['VK', 'HT', 'FT'])
    writer.writerows(rows.tolist())
    return stream.getvalue()


def decode_csv(text, expected_keys):
    """Reject dropped/reordered events rather than silently aligning by position."""
    reader = csv.reader(io.StringIO(text))
    if next(reader, None) != ['VK', 'HT', 'FT']:
        raise ValueError('Expected exact VK,HT,FT header')
    rows = []
    for fields in reader:
        if len(fields) != 3 or any(re.fullmatch(r'[+-]?[0-9]+', v) is None for v in fields):
            raise ValueError('Malformed integer row')
        rows.append([int(v) for v in fields])
    values = integer_rows(rows)
    keys = np.asarray(expected_keys)
    if keys.ndim != 1 or not np.array_equal(values[:, 0], keys):
        raise ValueError('Synthesized key sequence or event count changed')
    return values


def raw_residual_seconds(observed_rows, synthesized_rows):
    """Observed minus synthesized timings in seconds, before any policy cleanup.

Negative sentinels remain visible. This is the arithmetic primitive only,
not a claim about the paper's clipping order or final five model channels.
"""
    observed = integer_rows(observed_rows)
    synthesized = integer_rows(synthesized_rows)
    if observed.shape != synthesized.shape or not np.array_equal(observed[:, 0], synthesized[:, 0]):
        raise ValueError('Observed and synthesized event keys must align exactly')
    return (observed[:, 1:] - synthesized[:, 1:]).astype(np.float64) / 1000.0


def pad_features(features, *, length):
    """Zero-pad after feature construction; never send padding to the synthesizer."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError('Expected nonempty finite feature matrix')
    if isinstance(length, bool) or not isinstance(length, int) or length < len(features):
        raise ValueError('Padding length must be an integer at least the event count')
    return np.pad(features, ((0, length - len(features)), (0, 0)))
