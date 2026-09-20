"""Explicit calibrated lengths for padded capture-worker requests."""
import numpy as np

from type2branch_wire import validate_features

ANCHORS = (25, 50, 75, 100)


def validate_request(request):
    """Return id, float32 windows and integer lengths without inferring padding."""
    if not isinstance(request, dict) or set(request) != {'id', 'features', 'true_lengths'}:
        raise ValueError('Expected exactly id, features and true_lengths')
    identifier = request['id']
    if type(identifier) is not int or identifier < 0:
        raise ValueError('Request id must be a nonnegative integer')
    features = validate_features(request['features'])
    lengths = request['true_lengths']
    if (not isinstance(lengths, list) or len(lengths) != len(features)
            or any(type(length) is not int or length not in ANCHORS for length in lengths)):
        raise ValueError('One explicit integer length25/50/75/100 required per window')
    # The original validator has checked types, shape and finiteness. Inspect
    # original values, not its float32 output: small nonzero tails can underflow.
    original = np.asarray(request['features'])
    for index, length in enumerate(lengths):
        if np.any(original[index, length:, :] != 0):
            raise ValueError('All five padded-tail channels must be exactly zero')
    return identifier, features, np.asarray(lengths, dtype=np.int64)
