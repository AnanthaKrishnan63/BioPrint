import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from hmog_tap_paired_train import split_windows,paired_scores,ACCOUNTS,PATH_ARGUMENTS,canonical_paths,require_bound_paths

class PairedTrainTests(unittest.TestCase):
    def test_tampered_cli_path_rejected_before_artifact_access(self):
        args=SimpleNamespace(**{name:Path('synthetic')/name for name in PATH_ARGUMENTS})
        plan={'canonical_paths':canonical_paths(args)}
        require_bound_paths(args,plan)
        for name in PATH_ARGUMENTS:
            changed=SimpleNamespace(**vars(args));setattr(changed,name,Path('other')/name)
            with self.assertRaises(ValueError):require_bound_paths(changed,plan)
    def fixture(self):
        subjects=np.repeat(np.array(ACCOUNTS),3);sessions=np.tile([1,2,9],4)
        meta={'subject':subjects,'session':sessions}
        rng=np.random.default_rng(41)
        taps=np.vstack([rng.uniform(1,2,(80,11))+j//3 for j in range(12)])
        index=np.repeat(np.arange(12),80)
        base=np.vstack([np.full(64,j//3+1.)+rng.normal(0,.01,64) for j in range(12)])
        return meta,taps,index,{b:base.copy() for b in ('key','imu','joint')}
    def test_split_is_session_disjoint(self):
        meta,taps,index,z=self.fixture()
        g,c,d,chosen,missing=split_windows(meta['subject'],meta['session'],np.ones(12,dtype=bool))
        self.assertEqual((g.sum(),c.sum(),d.sum()),(4,4,4))
        self.assertFalse((g&c).any() or (g&d).any() or (c&d).any())
        self.assertEqual(set(chosen.values()),{1});self.assertEqual(missing,[])
    def test_same_windows_fusion_and_complete_methods(self):
        meta,taps,index,z=self.fixture();r,a=paired_scores(z,meta,taps,index)
        self.assertEqual(r['status'],'complete')
        self.assertEqual(len(r['metrics']),6)
        self.assertEqual(a['distance_joint_tap11'].shape,(12,4))
        self.assertEqual(r['counts']['gallery_windows'],4)
    def test_calibration_rows_do_not_fit_fusion_scales(self):
        meta,taps,index,z=self.fixture();before,_=paired_scores(z,meta,taps,index)
        for values in z.values():values[meta['session']==2]+=100
        after,_=paired_scores(z,meta,taps,index)
        self.assertEqual(before['scales'],after['scales'])
    def test_missing_gallery_is_not_dropped(self):
        meta,taps,index,z=self.fixture();keep=~np.isin(index,[0,1])
        r,a=paired_scores(z,meta,taps[keep],index[keep])
        self.assertEqual(r['status'],'infeasible');self.assertIsNone(a)

if __name__=='__main__':unittest.main()
