import struct
import unittest
from brainrun_archive_metadata import parse_directory


class DirectoryTests(unittest.TestCase):
    def entry(self):
        name=b'gestures.json'
        return struct.pack('<4s6H3L5H2L',b'PK\x01\x02',20,20,0,8,0,0,123,10,20,len(name),0,0,0,0,0,7)+name

    def test_metadata_only_fields(self):
        row=parse_directory(self.entry(),1)[0]
        self.assertEqual((row['name'],row['compressed_bytes'],row['uncompressed_bytes'],row['local_header_offset']),('gestures.json',10,20,7))

    def test_incomplete_or_wrong_count(self):
        for data,count in [(self.entry()[:-1],1),(self.entry(),2),(b'notzip',1)]:
            with self.assertRaises(ValueError):parse_directory(data,count)


if __name__=='__main__':unittest.main()
