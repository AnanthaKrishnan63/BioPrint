import io
import struct
import pytest
from brainrun_guarded_stream import canonical, iter_authorized
from brainrun_bson_metadata import MAX_DOCUMENT_BYTES


def string(value):
    encoded = value.encode()
    return struct.pack('<i',len(encoded)+1)+encoded+b'\0'


def record(device, user=None, poison=False):
    fields = b'\x02device_id\0'+string(device)
    if user is not None:fields += b'\x02user_id\0'+string(user)
    if poison:fields += b'\x03behavior\0'+struct.pack('<i',8)+b'\xff'*4
    fields += b'\0'
    return struct.pack('<i',len(fields)+4)+fields


def fixture():
    roles = {}
    mapping = {}
    for name, role, active in [('yes','fit',True),('other','fit',True),('inactive','fit',False),
                               ('select','selection',True),('cal','calibration',True),
                               ('dev','dev',True),('sealed','test',False)]:
        user=canonical(('string',name))
        roles[user]={'identity_role':role,'active':active,
                     'partition':'train' if role in ['fit','selection','calibration'] else role}
        mapping[canonical(('string','device-'+name))]=user
    return mapping, roles, {canonical(('string','yes'))}


def test_all_unauthorized_roles_opaque_to_decoder():
    mapping, roles, allowed=fixture(); calls=[]
    raw=b''.join(record('device-'+name,name,poison=True) for name in
                 ['sealed','dev','select','cal','inactive','other'])+record('device-yes','yes')
    def decoder(value):
        assert b'\xff' not in value
        calls.append(value)
        return 'decoded'
    assert list(iter_authorized(io.BytesIO(raw),mapping,roles,allowed,decoder)) == [(next(iter(allowed)),'decoded')]
    assert len(calls)==1


@pytest.mark.parametrize('name',['other','inactive','select','cal','dev','sealed','unknown'])
def test_only_explicit_active_fit_allowlist(name):
    mapping, roles, allowed=fixture()
    # 'other' is legitimately active fit, so excluding all mapped devices is the failure here.
    if name=='other':mapping={k:v for k,v in mapping.items() if v!=canonical(('string',name))}
    with pytest.raises(ValueError):
        list(iter_authorized(io.BytesIO(b''),mapping,roles,{canonical(('string',name))},lambda _:None))


@pytest.mark.parametrize('raw',[record('missing'),record('device-yes','wrong'),record('device-sealed','wrong'),
                               struct.pack('<i',5)+b'\0'])
def test_bad_identity_fails_before_decoder(raw):
    mapping,roles,allowed=fixture()
    with pytest.raises(ValueError):
        list(iter_authorized(io.BytesIO(raw),mapping,roles,allowed,lambda _:pytest.fail('Decoder called')))


def test_cap_does_not_even_frame_poison_tail():
    mapping,roles,allowed=fixture(); first=record('device-yes'); stream=io.BytesIO(first+b'bad-tail')
    assert len(list(iter_authorized(stream,mapping,roles,allowed,lambda _:True,max_selected_records=1)))==1
    assert stream.tell()==len(first)


@pytest.mark.parametrize('cap',[0,-1,True,1.5])
def test_invalid_cap(cap):
    mapping,roles,allowed=fixture()
    with pytest.raises(ValueError):list(iter_authorized(io.BytesIO(),mapping,roles,allowed,lambda _:None,max_selected_records=cap))


@pytest.mark.parametrize('payload',[b'\x01',struct.pack('<i',MAX_DOCUMENT_BYTES+1),struct.pack('<i',20)+b'abc'])
def test_document_bounds_and_truncation(payload):
    mapping,roles,allowed=fixture()
    with pytest.raises(ValueError):list(iter_authorized(io.BytesIO(payload),mapping,roles,allowed,lambda _:pytest.fail('Decoder called')))


def test_typed_device_collision_does_not_join():
    mapping,roles,allowed=fixture()
    mapping={canonical(('int32',12)):next(iter(allowed))}
    with pytest.raises(ValueError,match='Unknown record device'):
        list(iter_authorized(io.BytesIO(record('12')),mapping,roles,allowed,lambda _:pytest.fail('Decoder called')))


def test_empty_allowlist_fails_before_stream_access():
    mapping,roles,_=fixture()
    class Unreadable:
        def read(self,n):pytest.fail('Stream accessed')
    with pytest.raises(ValueError):list(iter_authorized(Unreadable(),mapping,roles,set(),lambda _:None))


def test_changed_authorization_between_yields_rejected():
    mapping,roles,allowed=fixture(); user=next(iter(allowed)); raw=record('device-yes')
    iterator=iter_authorized(io.BytesIO(raw+raw),mapping,roles,allowed,lambda _:True)
    assert next(iterator)==(user,True)
    roles[user]['identity_role']='test'
    with pytest.raises(ValueError):next(iterator)


def test_per_user_cap_skips_saturated_user_and_stops_before_tail():
    mapping,roles,allowed=fixture();allowed.add(canonical(('string','other')))
    first=record('device-yes'); opaque=record('device-yes',poison=True); other=record('device-other')
    prefix=first+opaque+other;stream=io.BytesIO(prefix+b'bad-tail')
    def decoder(value):
        assert b'\xff' not in value
        return True
    result=list(iter_authorized(stream,mapping,roles,allowed,decoder,max_per_user=1))
    assert [r[0] for r in result]==[canonical(('string','yes')),canonical(('string','other'))]
    assert stream.tell()==len(prefix)


def test_total_cap_stops_before_unfilled_user_caps():
    mapping,roles,allowed=fixture();allowed.add(canonical(('string','other')))
    first=record('device-yes');stream=io.BytesIO(first+b'bad-tail')
    assert len(list(iter_authorized(stream,mapping,roles,allowed,lambda _:True,
                                   max_per_user=4,max_selected_records=1)))==1
    assert stream.tell()==len(first)


@pytest.mark.parametrize('cap',[0,-1,True,1.5])
def test_invalid_per_user_cap(cap):
    mapping,roles,allowed=fixture()
    with pytest.raises(ValueError):list(iter_authorized(io.BytesIO(),mapping,roles,allowed,lambda _:None,max_per_user=cap))
