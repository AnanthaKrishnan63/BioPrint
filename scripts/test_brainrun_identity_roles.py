import unittest
from brainrun_identity_roles import assign,canonical,CAPS


class RoleTests(unittest.TestCase):
    def test_assignment_stable_disjoint_and_capped(self):
        users=[canonical(('string',str(i))) for i in range(3000)]
        roles=assign(users)
        self.assertEqual(roles,assign(reversed(users)))
        for role,cap in CAPS.items():
            self.assertEqual(sum(r['active'] and r['identity_role']==role for r in roles.values()),cap)
        for row in roles.values():
            if row['partition']=='test':self.assertFalse(row['active'])
            if not row['active']:self.assertEqual(row['behavioral_access'],'sealed')

    def test_identifier_type_distinctions(self):
        self.assertNotEqual(canonical(('string','1')),canonical(('int32',1)))


if __name__=='__main__':unittest.main()
