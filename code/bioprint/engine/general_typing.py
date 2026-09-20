"""Password-independent timing summaries and enrollment-to-query comparison.

This research matcher accepts arbitrary desktop key sequences. It does not
assert that arbitrary passwords have been validated or replace credentials.
"""
import numpy as np


def summarize(hold, down_down, up_down):
    channels = [np.asarray(c, dtype=float) for c in (hold, down_down, up_down)]
    if any(c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() for c in channels):
        raise ValueError('At least two finite timings per channel required')
    if (channels[0] < 0).any():
        raise ValueError('Negative hold duration')
    # Signed DD/UD are retained: some source recordings use release ordering.
    return np.concatenate([
        np.r_[np.quantile(c, [.1, .25, .5, .75, .9]), c.mean(), c.std()]
        for c in channels
    ])


def from_keystrokes(strokes):
    if len(strokes) < 3:
        raise ValueError('At least three timed keys required')
    return summarize([k.up - k.down for k in strokes],
                     [b.down - a.down for a, b in zip(strokes, strokes[1:])],
                     [b.down - a.up for a, b in zip(strokes, strokes[1:])])


def profile(enrollment):
    x = np.asarray(enrollment, dtype=float)
    if x.ndim != 2 or x.shape[1] != 21 or len(x) < 10 or not np.isfinite(x).all():
        raise ValueError('Ten finite 21-feature enrollment vectors required')
    center = np.median(x, axis=0)
    spread = np.maximum.reduce([np.mean(abs(x - center), axis=0),
                                np.full(21, 10.), abs(center) * .1])
    return center, spread


def comparison(enrollment_profile, queries):
    center, spread = enrollment_profile
    x = np.asarray(queries, dtype=float)
    if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all():
        raise ValueError('Expected finite 21-feature queries')
    z = np.clip((x - center) / spread, -100, 100)
    return np.concatenate([abs(z), z], axis=1)
