import unittest
from hmog_contact_parser import parse_contacts
from hmog_fit_feature_extract_v3 import strict_pairs,window_quality,gap_statistics
from test_hmog_fit_feature_extract_v2 import example

class RetainedPairTests(unittest.TestCase):
    def test_invalid_pointer_counts_excluded(self):
        for count in [0,-1,1.5,float('nan')]:
            k,t,s,a=example();t[0][3]=count
            self.assertEqual(len(parse_contacts(t)[0]),51)
    def test_gaps_reported_without_fabrication(self):
        k,t,s,a=example();k.append([250,250,7,1,43,0]);p=strict_pairs(k,t)
        self.assertEqual(len(p),52)
        self.assertEqual(window_quality(k,t,p,s,s,0,30000,a)[0],[])
        g=gap_statistics(k,p)
        self.assertEqual(g['orphan_or_ambiguous_extra_events'],1)
        self.assertAlmostEqual(g['retained_endpoint_ratio'],104/105)
    def test_cancel_still_removes_used_contact(self):
        k,t,s,a=example();t.append([50,50,7,1,0,3,1,1,1,1,0])
        self.assertEqual(len(strict_pairs(k,t)),51)
    def test_unrelated_cancel_does_not_remove_valid_window(self):
        k,t,s,a=example();t.append([28000,28000,7,1,0,3,1,1,1,1,0])
        self.assertEqual(window_quality(k,t,strict_pairs(k,t),s,s,0,30000,a)[0],[])

if __name__=='__main__':unittest.main()
