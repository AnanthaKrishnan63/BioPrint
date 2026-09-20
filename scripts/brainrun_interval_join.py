"""Unique temporal containment within exact typed user/device/task contexts.

Callers must establish common millisecond units and map screens to task labels
before this helper. It never guesses units, task aliases, or nearest intervals.
"""
from bisect import bisect_right
from collections import defaultdict
from numbers import Integral


def _identity(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError('Identity must be an immutable (type_tag,value) pair')
    tag, item = value
    valid = ((tag == 'string' and isinstance(item, str))
             or (tag == 'objectid' and isinstance(item, str) and len(item) == 24
                 and all(c in '0123456789abcdef' for c in item))
             or (tag == 'int32' and type(item) is int and -(2**31) <= item < 2**31)
             or (tag == 'int64' and type(item) is int and -(2**63) <= item < 2**63))
    if not valid:
        raise ValueError('Invalid typed identity')
    return value


def _interval(row, *, game):
    start, stop = row['start_ms'], row['stop_ms']
    if (isinstance(start, bool) or isinstance(stop, bool)
            or not isinstance(start, Integral) or not isinstance(stop, Integral)
            or start < 0 or stop < start or (game and stop == start)):
        raise ValueError('Nonnegative integer-ms interval with valid duration required')
    return int(start), int(stop)


def _key(row):
    task = row['task']
    if not isinstance(task, str) or not task:
        raise ValueError('Explicit nonempty task label required')
    return _identity(row['user']), _identity(row['device']), task


class IntervalJoin:
    """Index immutable copies of game keys, intervals, and typed game IDs."""
    def __init__(self, games):
        grouped = defaultdict(list)
        seen = set()
        for game in games:
            game_id = _identity(game['id'])
            if game_id in seen:
                raise ValueError('Duplicate game identity')
            seen.add(game_id)
            start, stop = _interval(game, game=True)
            grouped[_key(game)].append((start, stop, game_id))
        self._groups = {}
        for key, records in grouped.items():
            records.sort(key=lambda row: row[0])
            self._groups[key] = (tuple(row[0] for row in records), tuple(records))

    def match(self, gesture):
        start, stop = _interval(gesture, game=False)
        key = _key(gesture)
        starts, records = self._groups.get(key, ((), ()))
        candidates = tuple(game_id for _, game_stop, game_id in records[:bisect_right(starts, start)]
                           if stop <= game_stop)
        if not candidates:
            return {'status': 'no_game', 'game_id': None, 'candidate_count': 0}
        if len(candidates) > 1:
            return {'status': 'ambiguous', 'game_id': None, 'candidate_count': len(candidates)}
        return {'status': 'matched', 'game_id': candidates[0], 'candidate_count': 1}


def join_intervals(games, gestures):
    """Return one explicit result per gesture in original gesture order."""
    index = IntervalJoin(games)
    return [index.match(gesture) for gesture in gestures]
