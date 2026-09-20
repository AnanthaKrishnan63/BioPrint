import io
import struct
import pytest
from brainrun_bson_metadata import extract_metadata, iter_metadata, MAX_DOCUMENT_BYTES


def i32(value): return struct.pack('<i', value)
def string(value): return i32(len(value) + 1) + value + b'\0'
def element(kind, name, value): return bytes([kind]) + name + b'\0' + value
def document(*elements):
    body = b''.join(elements) + b'\0'
    return i32(len(body) + 4) + body


def test_identity_types_and_custom_allowlist():
    raw = document(element(2,b'device_id',string(b'12')), element(7,b'_id',bytes(range(12))),
                   element(16,b'user_id',i32(12)), element(18,b'custom',struct.pack('<q',12)))
    result = extract_metadata(raw, fields=['device_id','_id','user_id','custom'])
    assert result == {'device_id':('string','12'), '_id':('objectid',bytes(range(12)).hex()),
                      'user_id':('int32',12), 'custom':('int64',12)}
    assert len(set(result.values())) == 4


@pytest.mark.parametrize('kind,payload', [
    (1,struct.pack('<d',float('nan'))), (2,string(b'\xff')),
    (3,i32(8)+b'\xff\xff\xff\xff'), (4,i32(8)+b'\xff\xff\xff\xff'),
    (5,i32(3)+b'\x02'+b'\xff'*3), (6,b''), (7,b'\xff'*12), (8,b'\xff'),
    (9,b'\xff'*8), (10,b''), (11,b'\xff\0\xff\0'),
    (12,string(b'\xff')+b'\xff'*12), (13,string(b'\xff')), (14,string(b'\xff')),
    (15,i32(14)+b'\xff'*10), (16,b'\xff'*4), (17,b'\xff'*8), (18,b'\xff'*8),
    (19,b'\xff'*16), (127,b''), (255,b'')])
def test_all_standard_types_skip_opaque_behavior(kind,payload):
    raw = document(element(kind,b'behavior',payload),element(2,b'device_id',string(b'safe')))
    assert extract_metadata(raw) == {'device_id':('string','safe')}


def test_nested_identity_and_malformed_nested_names_never_decoded():
    nested = document(element(2,b'\xff',string(b'\xff')),element(2,b'user_id',string(b'hidden')))
    assert extract_metadata(document(element(3,b'data',nested))) == {}


@pytest.mark.parametrize('raw', [
    document(element(2,b'\xff',string(b'opaque'))),
    document(element(2,b'_id',string(b'\xff'))),
    document(element(1,b'_id',struct.pack('<d',1.))),
    document(element(16,b'unknown',i32(1)),element(16,b'unknown',i32(2))),
    document(element(16,b'_id',i32(1)),element(16,b'_id',i32(2))),
    document(element(20,b'unknown',b'')), document(element(0,b'unknown',b'')),
    document(element(2,b'x',i32(-1))), document(element(3,b'x',i32(4))),
    document(element(5,b'x',i32(-1)+b'\0')),
    document(element(2,b'x',i32(2)+b'aa')), document(element(11,b'x',b'unterminated')),
    document(element(16,b'x',b'\0')), document(element(3,b'x',i32(100))),
])
def test_invalid_top_level_framing_or_metadata_rejected(raw):
    with pytest.raises(ValueError):extract_metadata(raw)


def test_concatenation_and_partial_reads():
    a=document(element(16,b'_id',i32(3))); b=document(element(2,b'_id',string(b'z')))
    class Short(io.BytesIO):
        def read(self,n=-1):return super().read(min(n,2))
    assert list(iter_metadata(Short(a+b))) == [{'_id':('int32',3)},{'_id':('string','z')}]
    assert list(iter_metadata(io.BytesIO(b''))) == []


def test_every_truncated_prefix_fails():
    raw=document(element(2,b'_id',string(b'identity')))
    for stop in range(1,len(raw)):
        with pytest.raises(ValueError):list(iter_metadata(io.BytesIO(raw[:stop])))
    with pytest.raises(ValueError):list(iter_metadata(io.BytesIO(raw+b'\x01')))


@pytest.mark.parametrize('size',[0,4,-1,MAX_DOCUMENT_BYTES+1])
def test_size_rejected_before_payload_read(size):
    class PrefixOnly:
        calls=0
        def read(self,n):
            self.calls+=1
            assert self.calls==1 and n==4
            return i32(size)
    with pytest.raises(ValueError):list(iter_metadata(PrefixOnly()))


def test_exact_length_and_terminal_required():
    raw=document()
    for altered in [raw+b'\0',raw[:-1]+b'1',i32(6)+b'\0',b'']:
        with pytest.raises(ValueError):extract_metadata(altered)
    assert extract_metadata(raw)=={}


@pytest.mark.parametrize('fields',[['_id','_id'], '_id', [''], ['x\0y']])
def test_bad_allowlist_rejected(fields):
    with pytest.raises(ValueError):extract_metadata(document(),fields)
