"""External official seeded vectors plus generated edge-case checks."""
import hashlib
import json
from pathlib import Path
import re
import unittest

from type2branch_random import LegacyRandom, first_synthesis_thread, fill_average_fallback

REF = Path(__file__).resolve().parents[1] / 'research/benchmarks/references/type2branch_random'


class RandomTests(unittest.TestCase):
    def test_fallback_feature_order_and_stream_continuity(self):
        import numpy as np
        class Draws:
            def __init__(self):
                self.values = iter([0.1, 0.2, 0.3, 0.4])
            def next_double(self):
                return next(self.values)
        rng = Draws()
        original = np.array([[np.nan, np.nan], [7, np.nan]])
        np.testing.assert_array_equal(fill_average_fallback(original, rng),
                                      [[100, 200], [7, 300]])
        np.testing.assert_array_equal(fill_average_fallback([[8, np.nan]], rng), [[8, 400]])
        self.assertTrue(np.isnan(original[0, 0]))
        for invalid in [[[np.inf, 0]], [[-1, 0]], [[1.5, 0]], [[1501, 0]], [1, 2]]:
            with self.assertRaises(ValueError):
                fill_average_fallback(invalid, rng)

    def test_pinned_official_vectors(self):
        receipts = json.loads((REF / 'receipts.json').read_text())
        for receipt in receipts:
            self.assertEqual(hashlib.sha256((REF / receipt['file']).read_bytes()).hexdigest(),
                             receipt['sha256'])
        source = (REF / 'tests.cs').read_text()
        block = source.split('public void ExpectedValues(bool derived)', 1)[1].split(
            'public void ExpectedValues_Next64', 1)[0]
        vectors = re.findall(r'new int\[\]\s*\{([^}]+)\}', block)
        self.assertEqual(len(vectors), 20)
        for seed, vector in enumerate(vectors):
            expected = [int(x.strip()) for x in vector.split(',') if x.strip()]
            self.assertEqual(len(expected), 10)
            rng = LegacyRandom(seed)
            self.assertEqual([rng.next() for _ in expected], expected)
            doubles = LegacyRandom(seed)
            self.assertEqual([doubles.next_double() for _ in expected],
                             [x * (1.0 / (2**31 - 1)) for x in expected])

    def test_signed_seed_and_extremes(self):
        for a, b in [(1234, -1234), (2**31 - 1, -2**31)]:
            left, right = LegacyRandom(a), LegacyRandom(b)
            for _ in range(1000):
                value = left.next()
                self.assertEqual(value, right.next())
                self.assertTrue(0 <= value < 2**31 - 1)
        for seed in [-2**31 - 1, 2**31]:
            with self.assertRaises(ValueError):
                LegacyRandom(seed)
        with self.assertRaises(TypeError):
            LegacyRandom(1.5)

    def test_thread_seed_is_indirect(self):
        direct = LegacyRandom(1234)
        child_seed = direct.next()
        expected = LegacyRandom(child_seed)
        actual = first_synthesis_thread()
        self.assertEqual([actual.next() for _ in range(100)],
                         [expected.next() for _ in range(100)])


if __name__ == '__main__':
    unittest.main()
