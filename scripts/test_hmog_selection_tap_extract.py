import unittest
import numpy as np
from hmog_selection_tap_extract import validate_metadata

class SelectionTapMetadataTests(unittest.TestCase):
    def sample(self):
        return {'subject':np.array(['180679','622852']), 'session':np.array([3,9]),
                'role':np.array(['train_enrollment','train_selection']), 'activity_id':np.array([1,2]),
                'window_index':np.array([0,0])}
    def test_exact_selection_metadata(self):
        m=self.sample();self.assertIs(validate_metadata(m,2),m)
    def test_other_cohort_or_test_denied(self):
        for name,value in [('subject','717868'),('subject','556357'),('session',17),('role','train_fit')]:
            m=self.sample();m[name][0]=value
            with self.assertRaises(PermissionError):validate_metadata(m,2)
    def test_duplicate_and_nonintegral_windows_denied(self):
        m=self.sample()
        for a in m.values():a[1]=a[0]
        with self.assertRaises(ValueError):validate_metadata(m,2)
        m=self.sample();m['window_index']=np.array([0.,1.])
        with self.assertRaises(ValueError):validate_metadata(m,2)
    def test_frozen_sixty_row_requirement(self):
        with self.assertRaises(ValueError):validate_metadata(self.sample())

if __name__=='__main__':unittest.main()
