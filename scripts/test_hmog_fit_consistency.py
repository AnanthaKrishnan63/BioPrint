import unittest
from hmog_fit_consistency_audit import coverage

class CoverageTests(unittest.TestCase):
    def test_membership_time_and_orientation_separate(self):
        activities={7:[7,1,1,100,200,10,20,0,3]}
        rows=[[100,15,7,0,42,0],[101,21,7,1,42,0],[102,15,8,0,42,0],
              [103,15,7,0,42,1]]
        r=coverage(rows,activities,5)
        self.assertEqual(r['portrait_writing_in_bounds_rows'],1)
        self.assertEqual(r['unknown_activity_rows'],1)
        self.assertEqual(r['outside_activity_relative_bounds'],1)
        self.assertFalse(r['activity_membership_pass'])
        self.assertFalse(r['relative_bounds_pass'])

if __name__=='__main__':unittest.main()
