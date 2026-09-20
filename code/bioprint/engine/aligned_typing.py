"""Shared matcher features that retain alignment to each account's password."""
import numpy as np


def comparison(model, queries):
    x = np.asarray(queries, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model.names) or not np.isfinite(x).all():
        raise ValueError('Query must match the personal enrollment feature schema')
    if not np.isfinite(model.threshold) or model.threshold <= 0:
        raise ValueError('Positive finite personal threshold required')
    signed = np.clip((x - model.center) / model.spread, -6, 6)
    output = []
    for family in ('H.', 'DD.', 'UD.'):
        indices = [i for i, name in enumerate(model.names) if name.startswith(family)]
        if not indices:
            raise ValueError('Hold, DD and UD features required')
        z = signed[:, indices]
        a = abs(z)
        output.extend([*np.quantile(a, [.1, .25, .5, .75, .9], axis=1),
                       a.mean(1), a.std(1), np.median(z, axis=1), z.mean(1), z.std(1)])
    output.append(abs(signed).mean(1) / model.threshold)
    return np.column_stack(output)
