import unittest
from unittest.mock import patch
import numpy as np
from hmog_dev_tap_extract import validate_metadata, guarded_rows

class DevTapMetadataTests(unittest.TestCase):
    def sample(self):
        return {'subject':np.array(['556357','219303','777078','737973','219303','777078']),
                'session':np.array([8,1,1,1,10,9]),
                'role':np.array(['train_support']*4+['dev_probe']*2),
                'activity_id':np.arange(6), 'window_index':np.zeros(6,dtype=int)}
    def test_four_accounts_two_without_probes_allowed(self):
        m=self.sample();self.assertIs(validate_metadata(m,6),m)
    def test_other_cohort_test_or_wrong_role_denied(self):
        for key,value in [('subject','717868'),('session',17),('role','train_fit')]:
            m=self.sample();m[key][0]=value
            with self.assertRaises(PermissionError):validate_metadata(m,6)
    def test_missing_account_denied(self):
        m=self.sample();m['subject'][0]='219303'
        with self.assertRaises(PermissionError):validate_metadata(m,6)
    def test_duplicate_and_noninteger_denied(self):
        m=self.sample()
        for v in m.values():v[4]=v[1]
        with self.assertRaises(ValueError):validate_metadata(m,6)
        m=self.sample();m['session']=m['session'].astype(float)
        with self.assertRaises(ValueError):validate_metadata(m,6)
    def test_exact_sixty_eight_rows_required(self):
        with self.assertRaises(ValueError):validate_metadata(self.sample())
    def test_exact_guard_scope(self):
        class Reader:
            def open_member(self,*a,**kw):
                self.args=a;self.kw=kw;raise RuntimeError('sentinel')
        r=Reader()
        with self.assertRaises(RuntimeError):list(guarded_rows(r,'556357',8,'TouchEvent_im.csv'))
        self.assertEqual(r.kw,dict(expected_cohort='dev',expected_role='train_support',purpose='tap_dev_evaluation_feature_extraction'))

if __name__=='__main__':unittest.main()
