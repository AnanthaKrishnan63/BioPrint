import unittest
from hmog_clock_audit import complete_key_pairs

class KeyPairAuditTests(unittest.TestCase):
    def test_logging_order_does_not_replace_event_time(self):
        rows = [[1001, 20, 7, 1, 42, 0], [1002, 10, 7, 0, 42, 0]]
        pairs, dropped = complete_key_pairs(rows)
        self.assertEqual(pairs, [(10, 20, 7, 42, 0)])
        self.assertEqual(dropped['missing_up'], 0)

    def test_duplicate_and_missing_cycles_are_dropped(self):
        rows = [[1, 10, 7, 0, 42, 0], [2, 10, 7, 0, 42, 0], [3, 20, 7, 1, 42, 0],
                [4, 30, 7, 0, 42, 0], [5, 40, 7, 1, 43, 0]]
        pairs, dropped = complete_key_pairs(rows)
        self.assertEqual(pairs, [])
        self.assertEqual(dropped['duplicate_events'], 1)
        self.assertEqual(dropped['missing_up'], 1)
        self.assertEqual(dropped['missing_down'], 1)

    def test_repeated_down_does_not_invent_release(self):
        rows = [[1, 10, 7, 0, 42, 0], [2, 15, 7, 0, 42, 0], [3, 20, 7, 1, 42, 0]]
        self.assertEqual(complete_key_pairs(rows)[0], [])

if __name__ == '__main__':
    unittest.main()
