import unittest
import numpy as np
from hmog_tap_windows import tap_window
from hmog_tap_reference import Conventions

C=Conventions('linear',0,0.0,'euclidean','ignore')
def r(t,action,p=0,count=1,x=0):return [t,t,7,count,p,action,x,0,1,1,0]
def run(rows,start=0,end=1000):return tap_window(rows,session='synthetic',activity=7,start_ms=start,end_ms=end,conventions=C)
def clean():return [r(10,0),r(20,1),r(30,0,x=2),r(40,1,x=2)]

class TapWindowTests(unittest.TestCase):
    def test_clean_adjacent_real_contacts(self):
        q=run(clean());self.assertEqual(q['scorable_taps'],1)
        self.assertIsNone(q['scan_mean'])
        self.assertEqual(q['tap_vectors'].shape,(1,11))
        self.assertEqual(q['tap_vectors'][0,0],10)
        self.assertEqual(q['tap_vectors'][0,10],100)
    def test_scan_requires_five_valid_vectors(self):
        rows=[]
        for i in range(6):rows.extend([r(i*30+1,0),r(i*30+10,1)])
        q=run(rows)
        self.assertTrue(q['scan_scorable'])
        self.assertEqual(q['scan_mean'].shape,(11,))
    def test_observed_invalid_events_never_bridged(self):
        for action in [0,1,2,3,4,99]:
            q=run(clean()+[r(25,action,p=9)])
            self.assertEqual(q['scorable_taps'],0,action)
    def test_duplicate_and_repeated_down_never_bridged(self):
        for extra in [r(10,0),r(15,0)]:self.assertEqual(run(clean()+[extra])['scorable_taps'],0)
    def test_cancel_invalidates_contact_and_clean_pair_can_recover(self):
        rows=[r(1,0),r(2,3),r(3,1)]+clean()
        self.assertEqual(run(rows)['scorable_taps'],1)
    def test_previous_window_not_borrowed_and_right_edge_excluded(self):
        self.assertEqual(run(clean(),start=25)['scorable_taps'],0)
        self.assertEqual(run(clean(),end=40)['scorable_taps'],0)
    def test_overlapping_multi_pointer_not_serialized(self):
        rows=[r(10,0),r(12,5,p=1,count=2),r(18,6,p=1,count=2),r(20,1)]
        self.assertEqual(run(rows)['scorable_taps'],0)
    def test_wrong_activity_and_nonfinite_rejected(self):
        rows=clean();rows[0][2]=8
        with self.assertRaises(ValueError):run(rows)
        rows=clean();rows[0][6]=np.nan
        with self.assertRaises(ValueError):run(rows)

if __name__=='__main__':unittest.main()
