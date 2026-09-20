import importlib.util
import json
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).parent))
import hmog_tap_reference as m

C=m.Conventions('linear',0,0.0,'euclidean','ignore')

def contact(times=(1000,1010,1020,1030),sizes=(1,2,3,4),x=3,y=4,session='s'):
    return m.Contact(session,tuple(times),tuple([x]*len(times)),tuple([y]*len(times)),tuple(sizes))

class TapTests(unittest.TestCase):
    def test_exact_hand_computed_descriptors_and_units(self):
        prev=contact((0,10),(1,1),x=0,y=0)
        a=m.tap11(contact(),prev,conventions=C)
        np.testing.assert_allclose(a,[30,2.5,2.5,np.sqrt(1.25),1.75,2.5,3.25,1,1,4,5])
        self.assertEqual(a[2],a[5]) # Preserve published median/Q2 redundancy.
    def test_unscorable_missing_previous_and_invalid_contacts(self):
        for current,previous in [(contact(),None),(contact(),contact(session='other')),(contact((1000,1000),(1,1)),contact((0,10),(1,1))),(contact(sizes=(1,np.nan,3,4)),contact((0,10),(1,1))),(contact(),contact((990,1100),(1,1)))]:
            with self.subTest(current=current,previous=previous):
                with self.assertRaises(ValueError):m.tap11(current,previous,conventions=C)
    def test_mean_std_distance_and_serialization(self):
        rows=np.ones((3,11));rows[:,0]=[1,2,3];rows[:,1]=[2,4,6]
        p=m.fit_profile(rows,conventions=C,min_taps=3)
        self.assertEqual(p['active_feature_count'],2)
        scan=m.scan_mean([rows[0],rows[2]])
        self.assertEqual(m.scaled_manhattan(p,scan),0)
        self.assertAlmostEqual(m.scaled_manhattan(p,rows[2]),2/np.sqrt(2/3))
        restored=m.profile_from_json(m.profile_json(p))
        self.assertEqual(m.scaled_manhattan(p,rows[2]),m.scaled_manhattan(restored,rows[2]))
        self.assertEqual(restored,p)
    def test_no_spread_and_min_enrollment_no_verdict(self):
        for rows,n in [(np.ones((4,11)),2),(np.ones((1,11)),2)]:
            with self.assertRaises(ValueError):m.fit_profile(rows,conventions=C,min_taps=n)
    def test_bad_profiles_and_numeric_overflow_rejected(self):
        rows=np.arange(33,dtype=float).reshape(3,11)
        p=m.fit_profile(rows,conventions=C,min_taps=2)
        for key,value in [('mean',[float('nan')]*11),('spread',[-1.]*11),('active_feature_count',0),('score_direction','larger_is_genuine')]:
            malformed=dict(p);malformed[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):m.profile_json(malformed)
        with self.assertRaises(ValueError):m.profile_from_json('{"bad":NaN}')
        with self.assertRaises(ValueError):m.scan_mean(np.full((2,11),1e308))
        with self.assertRaises(ValueError):m.fit_profile(np.full((2,11),1e308),conventions=C,min_taps=2)
    def test_explicit_spread_cutoff_and_ddof(self):
        rows=np.ones((2,11));rows[:,0]=[0,2];rows[:,1]=[0,4]
        p=m.fit_profile(rows,conventions=m.Conventions('linear',1,1.5,'euclidean','ignore'),min_taps=2)
        self.assertEqual(p['active_feature_count'],1)
        self.assertAlmostEqual(p['spread'][0],np.sqrt(2))
        self.assertEqual(p['active'][0],False)

if __name__=='__main__':unittest.main()
