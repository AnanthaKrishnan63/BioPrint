import unittest
from hmog_contact_parser import parse_contacts

def row(t,p,a,o=0,count=1):return [t,t,7,count,p,a,1,1,1,1,o]

class ContactParserTests(unittest.TestCase):
    def test_multitouch_preserves_each_real_contact(self):
        rows=[row(1,0,0),row(2,1,5,count=2),row(3,1,6,count=2),row(4,0,1)]
        self.assertEqual(len(parse_contacts(rows)[0]),2)
    def test_cancel_invalidates_all_active(self):
        rows=[row(1,0,0),row(2,1,5,count=2),row(3,0,3),row(4,1,6),row(5,0,1)]
        self.assertEqual(parse_contacts(rows)[0],[])
    def test_rotation_invalidates_affected_contact(self):
        self.assertEqual(parse_contacts([row(1,0,0),row(2,0,2,1),row(3,0,1)])[0],[])
    def test_duplicate_or_repeated_down_invalidates(self):
        d=row(1,0,0)
        for mid in [d,row(2,0,0)]:
            self.assertEqual(parse_contacts([d,mid,row(3,0,1)])[0],[])
    def test_missing_events_not_fabricated(self):
        self.assertEqual(parse_contacts([row(1,0,1),row(2,1,0)])[0],[])
    def test_logging_order_not_event_time(self):
        self.assertEqual(len(parse_contacts([row(2,0,1),row(1,0,0)])[0]),1)

if __name__=='__main__':unittest.main()
