import unittest
from hmog_window_feasibility import bins_complete,matched_pairs,count_session

class WindowTests(unittest.TestCase):
    def test_half_open_bins_and_no_synthetic_fill(self):
        self.assertTrue(bins_complete([i*300 for i in range(100)],0)['complete'])
        self.assertFalse(bins_complete([i*300 for i in range(1,101)],0)['complete'])
        self.assertEqual(bins_complete([30000],0)['records'],0)

    def test_unique_contact_required(self):
        k=[[100,10,7,0,42,0],[200,20,7,1,42,0]]
        t=[[100,10,7,1,0,0,1,1,1,1,0],[200,20,7,1,0,1,1,1,1,1,0]]
        self.assertEqual(len(matched_pairs(k,t)[0]),1)
        self.assertEqual(len(matched_pairs(k,t+[t[0]])[0]),0)

    def test_exact_threshold_and_missing_sensor_bin(self):
        a=[[7,1,1,0,30000,0,30000,0,3]]
        keys=[];touch=[]
        for i in range(52):
            p=i*500+10;r=p+100
            keys.extend([[p,p,7,0,42,0],[r,r,7,1,42,0]])
            touch.extend([[p,p,7,1,0,0,1,1,1,1,0],[r,r,7,1,0,1,1,1,1,1,0]])
        sensor=[[i*300,0,7,0,0,0,0] for i in range(100)]
        self.assertEqual(count_session(a,keys,touch,sensor,sensor)['eligible_windows'],1)
        bad=count_session(a,keys,touch,sensor[:-1],sensor)
        self.assertEqual(bad['eligible_windows'],0)
        self.assertEqual(bad['exclusion_reasons_nonexclusive']['acc_empty_bins'],1)

if __name__=='__main__':unittest.main()
