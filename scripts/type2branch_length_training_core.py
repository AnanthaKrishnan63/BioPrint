"""Pure helpers for a TRAIN-only controlled prefix-length study."""
import numbers

import numpy as np

LENGTHS = (25, 50, 75, 100)


def augment_prefixes(batch, lengths):
    """Copy N x100x5 features and zero all channels beyond each prefix.

    This perturbs already-computed features; it does not rerun raw-event
    synthesis or imply that the author architecture masks padded positions.
    """
    try:
        values = np.asarray(batch)
        sizes = np.asarray(lengths)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Rectangular batch and integer lengths required') from error
    if (values.ndim != 3 or values.shape[1:] != (100, 5) or len(values) == 0
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
        raise ValueError('Nonempty finite real N x100x5 batch required')
    if (sizes.shape != (len(values),) or sizes.dtype.kind not in 'iu'
            or any(isinstance(item, (bool, np.bool_)) for item in lengths)
            or not np.isin(sizes, LENGTHS).all()):
        raise ValueError('One integer length25/50/75/100 required per row')
    output = values.copy()
    keep = np.arange(100)[None, :] < sizes[:, None]
    output[~keep] = 0
    return output


def selection_key(record):
    """Minimize selection FRR, then EER, then optimizer-update count."""
    try:
        rates = [record[name] for name in
                 ('mean_selection_frr_at_1pct', 'mean_selection_eer')]
        updates = record['updates']
    except (KeyError, TypeError) as error:
        raise ValueError('Selection metrics and update count required') from error
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, numbers.Real)
           or not np.isfinite(value) or not 0 <= value <= 1 for value in rates):
        raise ValueError('Selection rates must be finite numbers in[0,1]')
    if (isinstance(updates, (bool, np.bool_))
            or not isinstance(updates, numbers.Integral) or updates < 0):
        raise ValueError('Updates must be a nonnegative integer')
    return float(rates[0]), float(rates[1]), int(updates)


def tuple_state(value):
    """Restore nested tuples lost when Python Random state is JSON encoded."""
    if isinstance(value, (list, tuple)):
        return tuple(tuple_state(item) for item in value)
    return value
