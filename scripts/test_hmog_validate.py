import unittest
from unittest.mock import patch
import numpy as np
from hmog_validate import load_dev_arrays,coverage_from_report,evaluate_embeddings,DEV_IDS,BRANCHES,TARGETS

class FakeArchive:
    def __init__(self,subject,session,role):
        self.values={'subject':np.array([subject]),'session':np.array([session]),'role':np.array([role])}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def __getitem__(self,key):
        if key in ('key','imu'):raise AssertionError('Feature observed before unauthorized metadata rejected')
        return self.values[key]

class ValidationTests(unittest.TestCase):
    def test_metadata_rejected_before_feature_copy(self):
        for subject,session,role in [('717868',9,'dev_probe'),('556357',17,'dev_probe'),
                                     ('556357',1,'train_enrollment'),('556357',9,'train_support')]:
            with patch('hmog_validate.np.load',return_value=FakeArchive(subject,session,role)):
                with self.assertRaises(PermissionError):load_dev_arrays('unused')

    def fixture(self,missing=False):
        ids=sorted(DEV_IDS);subjects=np.array(ids+ids);roles=np.array(['train_support']*4+['dev_probe']*4)
        z=np.repeat(np.array([0,10,20,30,0,10,20,30])[:,None],64,axis=1).astype(float)
        if missing:subjects=subjects[1:];roles=roles[1:];z=z[1:]
        coverage={s:{'dev_probe':{'candidate_windows':2}} for s in ids}
        thresholds={b:{t:1. for t in TARGETS} for b in BRANCHES}
        return {b:z for b in BRANCHES},subjects,roles,coverage,thresholds

    def test_frozen_decisions_and_missing_window_coverage(self):
        result,arrays=evaluate_embeddings(*self.fixture())
        self.assertEqual(result['status'],'complete')
        r=result['metrics']['joint']['operating_points']['0.01']
        self.assertEqual(r['frr'],0);self.assertEqual(r['far'],0)
        self.assertEqual(r['missing_genuine'],4)
        self.assertEqual(r['genuine_not_accepted_rate'],.5)
        self.assertEqual(arrays['joint_distances'].shape,(4,4))

    def test_missing_gallery_not_silently_dropped(self):
        result,arrays=evaluate_embeddings(*self.fixture(True))
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(len(arrays['accounts']),4)
        self.assertFalse(arrays['joint_claim_available'][:,0].any())
        self.assertTrue(np.isnan(arrays['joint_profiles'][0]).all())
        self.assertEqual(result['missing'][0]['missing_role'],'train_support')

    def test_metadata_exclusion_preserves_unknown_count(self):
        ids=sorted(DEV_IDS);report={'subjects':{s:{} for s in ids},'excluded_sessions':{}}
        for s in ids:
            report['excluded_sessions'][s]={str(n):{'reason':'empty_keypress','basis':'archive_metadata',
                'candidate_windows':None,'eligible_windows':None,'activity_contents_inspected':False} for n in range(1,17)}
        coverage=coverage_from_report(report,np.array([],dtype='U6'),np.array([],dtype=int))
        self.assertIsNone(coverage[ids[0]]['dev_probe']['excluded_session_candidate_windows'])
        report['excluded_sessions'][ids[0]]['1']['candidate_windows']=0
        with self.assertRaises(ValueError):coverage_from_report(report,np.array([],dtype='U6'),np.array([],dtype=int))

if __name__=='__main__':unittest.main()
