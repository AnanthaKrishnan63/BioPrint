import hashlib
import json
import unittest
from hmog_later_feature_extract import STAGES,validate_upstream,session_inventory,strict_pairs
from hmog_fit_feature_extract_v3 import strict_pairs as original

class LaterExtractionTests(unittest.TestCase):
    def report(self,stage):
        r={'status':'complete','plan':{'model_protocol_sha256':'protocol','selected_checkpoint_sha256':'a'*64},
           'dev_accessed':False}
        if stage=='calibration':r.update(selection_only=True,selected_checkpoint_sha256='a'*64)
        else:r.update(global_model_fitted=False,thresholds_sha256='b'*64)
        raw=json.dumps(r).encode();return raw,hashlib.sha256(raw).hexdigest()
    def test_exact_upstream_required(self):
        for stage in STAGES:
            raw,digest=self.report(stage)
            validate_upstream(raw,digest,stage,'protocol')
            for badsha,badprotocol in [('wrong','protocol'),(digest,'other')]:
                with self.assertRaises(ValueError):validate_upstream(raw,badsha,stage,badprotocol)
            r=json.loads(raw);r['status']='infeasible';bad=json.dumps(r).encode()
            with self.assertRaises(ValueError):validate_upstream(bad,hashlib.sha256(bad).hexdigest(),stage,'protocol')
    def test_stages_have_disjoint_fixed_roles(self):
        c,d=STAGES['calibration'],STAGES['dev']
        self.assertFalse(set(c['ids'])&set(d['ids']))
        self.assertEqual(d['support_role'],'train_support')
        self.assertEqual(c['probe_role'],'train_calibration')
        self.assertEqual(d['probe_role'],'dev_probe')
    def test_exact_feature_functions_and_unknown_counts(self):
        self.assertIs(strict_pairs,original)
        selected,excluded=session_inventory({},['556357'])
        self.assertEqual(selected['556357'],[])
        self.assertEqual(len(excluded['556357']),16)
        self.assertTrue(all(x['candidate_windows'] is None for x in excluded['556357'].values()))

if __name__=='__main__':unittest.main()
