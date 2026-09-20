import unittest
import numpy as np
from beacon_type2branch_selection import selection_key


class SelectionTests(unittest.TestCase):
    def test_perfect_lower_scores_are_genuine(self):
        self.assertEqual(selection_key([1, 1, 0, 0], [0, 1, 2, 3], .1), (0., 0., .1))

    def test_ties_cannot_be_split_to_claim_low_far(self):
        self.assertEqual(selection_key([1, 0, 0], [1, 1, 1], .01), (1., .5, .01))

    def test_lower_c_breaks_equal_metric_ties(self):
        keys = [selection_key([1, 0], [0, 1], c) for c in [1., .1, .01]]
        self.assertEqual(min(keys)[2], .01)

    def test_invalid_inputs(self):
        for labels, scores, c in [([1], [0], .1), ([1,0], [0,np.nan], .1),
                                  ([1,0], [0], .1), ([1,2], [0,1], .1),
                                  ([1,0], [0,1], 10.)]:
            with self.assertRaises(ValueError):
                selection_key(labels, scores, c)


if __name__ == '__main__':
    unittest.main()
