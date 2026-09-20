import copy
import unittest
from brainrun_record_quarantine import extend


class QuarantineTests(unittest.TestCase):
    def fixtures(self):
        roles={'device_to_user':{'known':'owner'},'quarantine_device_ids':['shared'],'users':{'owner':{'active':True}},'counts':{'active_by_role':{'fit':1}}}
        r={'member_crc_verified_at_eof':True,'counts':{'records':3,'known':1,'quarantine':1,'unknown':1,'missing_device':0,'conflicting_user':0},'unknown_devices':{'orphan':{'records':1}},'unknown_device_count':1}
        return roles,{'status':'record_identifier_metadata_audit_complete','behavioral_observations_decoded':False,'collections':{'games':copy.deepcopy(r),'gestures':copy.deepcopy(r)}}

    def test_no_identity_reselection_or_input_mutation(self):
        roles,audit=self.fixtures();before=copy.deepcopy(roles);out=extend(roles,audit)
        self.assertEqual(roles,before);self.assertEqual(out['users'],roles['users'])
        self.assertEqual(out['quarantine_device_ids'],['orphan','shared'])
        self.assertEqual(out['device_to_user'],roles['device_to_user'])

    def test_incomplete_or_conflicting_audit_rejected(self):
        for field,value in [('missing_device',1),('conflicting_user',1),('records',4),('unknown',2)]:
            roles,audit=self.fixtures();audit['collections']['games']['counts'][field]=value
            with self.assertRaises(ValueError):extend(roles,audit)


if __name__=='__main__':unittest.main()
