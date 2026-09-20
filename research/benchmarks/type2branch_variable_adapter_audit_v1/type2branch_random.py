"""Legacy seeded System.Random translation for the single-thread synthesis candidate.

Based on Microsoft's MIT-licensed reference source, retained with its license
under research/benchmarks/references/type2branch_random. Only Next() and
NextDouble() are implemented. This is not a CLR runtime parity claim.
"""
import operator


def int32(value):
    return (value + 2**31) % 2**32 - 2**31


class LegacyRandom:
    MBIG = 2**31 - 1

    def __init__(self, seed):
        seed = operator.index(seed)
        if not -2**31 <= seed <= self.MBIG:
            raise ValueError('Seed must fit signed Int32')
        subtraction = self.MBIG if seed == -2**31 else abs(seed)
        mj = 161803398 - subtraction
        state = [0] * 56
        state[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            state[ii] = mk
            mk = int32(mj - mk)
            if mk < 0:
                mk = int32(mk + self.MBIG)
            mj = state[ii]
        for _ in range(4):
            for i in range(1, 56):
                state[i] = int32(state[i] - state[1 + (i + 30) % 55])
                if state[i] < 0:
                    state[i] = int32(state[i] + self.MBIG)
        self.state = state
        self.inext = 0
        self.inextp = 21

    def next(self):
        self.inext = self.inext + 1 if self.inext < 55 else 1
        self.inextp = self.inextp + 1 if self.inextp < 55 else 1
        value = int32(self.state[self.inext] - self.state[self.inextp])
        if value == self.MBIG:
            value -= 1
        if value < 0:
            value = int32(value + self.MBIG)
        self.state[self.inext] = value
        return value

    def next_double(self):
        return self.next() * (1.0 / self.MBIG)


def first_synthesis_thread():
    """Fresh process, first thread: global seed 1234 supplies the local seed.

    Caller owns stream lifetime and any earlier draws; this does not reproduce
    arbitrary thread scheduling or initialize a new stream for each sequence.
    """
    return LegacyRandom(LegacyRandom(1234).next())


def fill_average_fallback(values, rng):
    """Fill missing model values, HT positions first then FT, without cleanup.

    Preserve the caller's stream across sequences. NaN represents a missing
    context; observed models consume no random draws.
    """
    import numpy as np
    values = np.array(values, dtype=float, copy=True)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('Expected N x 2 HT/FT predictions')
    present = values[~np.isnan(values)]
    if not np.isfinite(present).all() or (present < 0).any() or (present > 1500).any():
        raise ValueError('Model values must be finite cleaned timings')
    if (present != np.trunc(present)).any():
        raise ValueError('Average predictions must already be truncated')
    for feature in range(2):
        for position in range(len(values)):
            if np.isnan(values[position, feature]):
                values[position, feature] = int(1000 * rng.next_double())
    return values.astype(np.int64)
