import unittest
from hmog_selection_feature_extract_v1 import session_inventory, strict_pairs
from hmog_fit_feature_extract_v3 import strict_pairs as frozen_pairs

class SelectionPreparationTests(unittest.TestCase):
    def test_missing_and_empty_are_unknown_window_counts(self):
        entries={'180679/180679_session_1/KeyPressEvent.csv':{'uncompressed_bytes':42},
                 '180679/180679_session_2/KeyPressEvent.csv':{'uncompressed_bytes':0}}
        selected,excluded=session_inventory(entries,['180679'])
        self.assertEqual(selected,{'180679':[1]})
        self.assertEqual(len(excluded['180679']),15)
        self.assertEqual(excluded['180679']['2']['reason'],'empty_keypress')
        self.assertTrue(all(v['candidate_windows'] is None for v in excluded['180679'].values()))
        self.assertTrue(all(not v['activity_contents_inspected'] for v in excluded['180679'].values()))
    def test_uses_exact_frozen_pair_function(self):
        self.assertIs(strict_pairs,frozen_pairs)

if __name__=='__main__':unittest.main()
