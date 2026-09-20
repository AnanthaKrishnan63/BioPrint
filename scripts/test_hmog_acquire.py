import importlib.util
import io
import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

spec = importlib.util.spec_from_file_location('hmog_acquire', pathlib.Path(__file__).with_name('hmog_acquire.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class AcquisitionTests(unittest.TestCase):
    def test_stream_roundtrip_and_no_overwrite(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as z: z.writestr('session/sealed.csv', 'synthetic-data\n')
        data = inner.getvalue(); c = zlib.compressobj(wbits=-15); packed = c.compress(data) + c.flush()
        name = b'public_dataset/123.zip'
        header = struct.pack('<4s5H3L2H', b'PK\x03\x04', 20, 0, 8, 0, 0, zlib.crc32(data), len(packed), len(data), len(name), 0)
        outer = header + name + packed
        row = {'name': name.decode(), 'local_header_offset': 0, 'compressed_bytes': len(packed), 'uncompressed_bytes': len(data), 'crc32': zlib.crc32(data), 'identity_role': 'fit'}
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); (root / 'raw').mkdir(); (root / 'receipts').mkdir()
            with patch.object(m, 'OUT', root), patch.object(m, 'disk_bytes', return_value=0), patch.object(m, 'response', side_effect=lambda a,b: io.BytesIO(outer[a:b+1])):
                m.fetch(row)
                self.assertEqual((root / 'raw/123.zip').read_bytes(), data)
                with self.assertRaises(FileExistsError): m.fetch(row)
    def test_cohort_partition_excludes_test(self):
        import json
        p = json.loads((m.OUT / 'preregistered_acquisition.json').read_text())
        roles = [x['identity_role'] for x in p['cohort']]
        self.assertEqual([roles.count(x) for x in ['fit','selection','calibration','dev']], [4,2,2,4])
        selected = {x['name'] for x in p['cohort']}
        self.assertFalse(selected & {x['name'] for x in p['sealed_test_never_fetch'] + p['unused_sealed_never_fetch']})
        self.assertLess(p['raw_inner_zip_bytes'] + 2000000, m.LIMIT)

if __name__ == '__main__': unittest.main()
