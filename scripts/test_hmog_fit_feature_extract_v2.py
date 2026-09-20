import unittest
from hmog_fit_feature_extract_v2 import strict_pairs,window_quality,no_intervening_bad_keys


def example():
    keys=[];touch=[]
    for i in range(52):
        p=i*500+10;r=p+100
        keys.extend([[p,p,7,0,42,0],[r,r,7,1,42,0]])
        touch.extend([[p,p,7,1,0,0,1,1,1,1,0],[r,r,7,1,0,1,1,1,1,1,0]])
    sensor=[[i*300,i*300*1000000,7,0,0,0,0] for i in range(100)]
    return keys,touch,sensor,[7,1,1,0,30000,0,30000,0,3]

class StrictExtractionTests(unittest.TestCase):
    def test_valid_contract(self):
        k,t,s,a=example();p=strict_pairs(k,t)
        self.assertTrue(no_intervening_bad_keys(k,p))
        self.assertEqual(window_quality(k,t,p,s,s,0,30000,a)[0],[])

    def test_cancel_move_rotation_and_multitouch_rejected(self):
        for action,count,orientation in [(3,1,0),(2,2,0),(2,1,1)]:
            k,t,s,a=example();t.append([50,50,7,count,0,action,1,1,1,1,orientation])
            self.assertTrue(window_quality(k,t,strict_pairs(k,t),s,s,0,30000,a)[0])

    def test_intervening_missing_pair_rejects_window(self):
        k,t,s,a=example();k.append([250,250,7,1,43,0])
        reasons=window_quality(k,t,strict_pairs(k,t),s,s,0,30000,a)[0]
        self.assertIn('invalid_intervening_key_event',reasons)

    def test_sensor_orientation_and_native_order(self):
        k,t,s,a=example();s[40][6]=1;s[41][1]=0
        reasons=window_quality(k,t,strict_pairs(k,t),s,s,0,30000,a)[0]
        self.assertIn('acc_nonportrait',reasons)
        self.assertIn('acc_native_clock_backstep',reasons)

if __name__=='__main__':unittest.main()
