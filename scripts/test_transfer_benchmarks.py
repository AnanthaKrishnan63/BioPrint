"""Leakage/representation invariants for public cognitive and bot benchmarks."""
import unittest
import numpy as np
import cognitive_benchmark as cognitive
import delbot_benchmark as delbot


class TransferBenchmarkTests(unittest.TestCase):
    def test_test_loaders_refuse_access(self):
        for module in [cognitive, delbot]:
            with self.assertRaises(ValueError):
                module.load('test')

    def test_three_condition_slope_ignores_middle(self):
        first = np.arange(30, dtype=float)
        changed = first.copy()
        changed[[5, 20]] += 100
        np.testing.assert_equal(cognitive.representation(first, 'condition_index_slope'),
                                cognitive.representation(changed, 'condition_index_slope'))
        self.assertFalse(np.array_equal(cognitive.representation(first, 'condition_means'),
                                        cognitive.representation(changed, 'condition_means')))

    def test_cognitive_pairs_use_later_session_probe(self):
        data = {'a': {1: np.zeros(30), 2: np.ones(30)},
                'b': {1: np.full(30, 2.), 2: np.full(30, 4.)}}
        x, y = cognitive.pairs(data, 'full_rt_accuracy_profile')
        np.testing.assert_equal(y, [1, 0, 0, 1])
        np.testing.assert_equal(x[:, 0], [1, 4, -1, 2])

    def test_bot_geometry_excludes_absolute_location_resolution(self):
        def sample(scale, tx, ty, resolution):
            rows = [f'resolution:{resolution}']
            rows += [f'{i * 10},Move,{tx + scale * i},{ty + scale * i * i}' for i in range(12)]
            return ('\n'.join(rows) + '\n').encode()
        a = delbot.features(sample(1, 0, 0, '1920,1080'))[0]
        b = delbot.features(sample(3, 400, -80, '800,600'))[0]
        np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
