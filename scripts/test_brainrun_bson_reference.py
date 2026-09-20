"""Generated BSON interoperability checks against the pinned PyMongo codec."""
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.research-brainrun-deps'))
from bson import BSON, ObjectId, Int64
from brainrun_bson_metadata import extract_metadata, iter_metadata


class ReferenceTests(unittest.TestCase):
    def test_real_codec_id_types_and_opaque_nested_arrays(self):
        examples=[('alice',('string','alice')),(17,('int32',17)),
            (Int64(17),('int64',17)),(ObjectId('0123456789abcdef01234567'),('objectid','0123456789abcdef01234567'))]
        for value,expected in examples:
            encoded=BSON.encode({'trace':[{'x':float('nan'),'nested':{'device_id':'not-the-owner'}}],
                'device_id':value,'user_id':'owner','behavior':b'\xff'*100})
            self.assertEqual(extract_metadata(encoded,fields=('device_id','user_id')),
                {'device_id':expected,'user_id':('string','owner')})

    def test_concatenated_reference_documents(self):
        records=[{'device_id':str(i),'user_id':'owner','points':list(range(i))} for i in range(10)]
        stream=io.BytesIO(b''.join(BSON.encode(v) for v in records))
        extracted=list(iter_metadata(stream,fields=('device_id','user_id')))
        self.assertEqual([v['device_id'][1] for v in extracted],[str(i) for i in range(10)])


if __name__=='__main__':unittest.main()
