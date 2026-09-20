"""Protocol and numerical correctness checks, using only synthetic inputs."""
import unittest
import numpy as np
from pointer_benchmark import features, load_entries, threshold_at_far, stats


class PointerBenchmarkTests(unittest.TestCase):
    def test_sealed_content_refused_before_path_access(self):
        with self.assertRaisesRegex(ValueError, 'Refusing test'):
            load_entries([{'split': 'test_sealed', 'path': 'must-not-exist'}])

    def test_threshold_respects_tied_impostor_scores(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([.2, .2, .8, .8, .9, .7])
        for target in [0, .01, .25, .5]:
            threshold = threshold_at_far(labels, scores, target)
            self.assertLessEqual(stats(labels, scores, threshold)['far'], target)

    def test_features_ignore_absolute_screen_position(self):
        t = np.arange(128) * .01
        trace = np.column_stack([t, np.arange(128), np.sin(t)])
        translated = trace.copy()
        translated[:, 1:] += [700, 200]
        np.testing.assert_allclose(features(trace), features(translated), atol=1e-7)
        self.assertEqual(len(features(trace)), 29)

    def test_duplicate_timestamps_do_not_produce_invalid_features(self):
        trace = np.column_stack([np.zeros(128), np.arange(128), np.arange(128)])
        self.assertIsNone(features(trace))

if __name__ == '__main__':
    unittest.main()
