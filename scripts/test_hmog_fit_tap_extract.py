import unittest
import numpy as np
from hmog_fit_tap_extract import validate_metadata

class TapExtractMetadataTests(unittest.TestCase):
    def sample(self):
        return {'subject':np.array(['717868','717868']), 'session':np.array([3,9]),
                'role':np.array(['train_enrollment','train_fit']), 'activity_id':np.array([1,2]),
                'window_index':np.array([0,0])}
    def test_exact_metadata_rows_preserved(self):
        m=self.sample();self.assertIs(validate_metadata(m,2),m)
    def test_heldout_and_test_rejected(self):
        for name,value in [('subject','556357'),('session',17),('role','train_support')]:
            m=self.sample();m[name][0]=value
            with self.assertRaises(PermissionError):validate_metadata(m,2)
    def test_duplicate_window_rejected(self):
        m=self.sample()
        for v in m.values():v[1]=v[0]
        with self.assertRaises(ValueError):validate_metadata(m,2)
    def test_nonintegral_or_changed_row_count_rejected(self):
        m=self.sample();m['window_index']=np.array([0.,1.])
        with self.assertRaises(ValueError):validate_metadata(m,2)
        with self.assertRaises(ValueError):validate_metadata(self.sample(),207)

if __name__=='__main__':unittest.main()
