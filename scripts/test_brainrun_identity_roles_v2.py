import io
import pytest
from brainrun_identity_roles import partition
from brainrun_identity_roles_v2 import assign_quarantined
from brainrun_guarded_stream_v2 import canonical, iter_authorized
from test_brainrun_guarded_stream import fixture, record


def test_quarantine_all_affected_users_before_active_cap():
    users = [canonical(('string', f'user{i}')) for i in range(100)]
    fit = sorted((u for u in users if partition(u)[0] == 'fit'), key=lambda u: partition(u)[1])
    affected = fit[:2]
    mapping = {canonical(('string', f'device{i}')): [u] for i,u in enumerate(users)}
    shared = canonical(('string', 'shared'))
    mapping[shared] = affected
    caps = dict(fit=2, selection=2, calibration=2, dev=2)
    result = assign_quarantined(mapping, caps)
    assert result['quarantine_device_ids'] == [shared]
    assert set(result['quarantine_user_ids']) == set(affected)
    assert shared not in result['device_to_user']
    assert {u for u,r in result['users'].items() if r['identity_role']=='fit' and r['active']} == set(fit[2:4])
    for user in affected:
        row=result['users'][user]
        assert row['active'] is False and row['inactive_reason']=='ambiguous_device_link'
        assert row['behavioral_access']=='sealed'
        assert row['assignment_sha256']==partition(user)[1]
        with pytest.raises(ValueError):
            list(iter_authorized(io.BytesIO(),result['device_to_user'],result['users'],{user},lambda _:True,
                                 quarantine_device_ids=result['quarantine_device_ids']))
    for user,row in result['users'].items():
        assert row['identity_role']==partition(user)[0]


def test_shared_device_never_decodes_unknown_device_still_fails():
    mapping,roles,allowed=fixture(); shared=canonical(('string','shared'))
    prefix=record('shared','unmappable-user',poison=True)+record('device-yes')
    def decoder(raw):
        assert b'\xff' not in raw
        return True
    assert len(list(iter_authorized(io.BytesIO(prefix),mapping,roles,allowed,decoder,
                                    quarantine_device_ids=[shared])))==1
    with pytest.raises(ValueError,match='Unknown record device'):
        list(iter_authorized(io.BytesIO(record('unknown')),mapping,roles,allowed,decoder,
                             quarantine_device_ids=[shared]))


def test_quarantine_disjoint_tagged_and_inactive_reason_enforced():
    mapping,roles,allowed=fixture()
    for quarantine in [[next(iter(mapping))], ['untagged'], 'bad']:
        with pytest.raises(ValueError):
            list(iter_authorized(io.BytesIO(),mapping,roles,allowed,lambda _:True,quarantine_device_ids=quarantine))
    roles[next(iter(allowed))]['inactive_reason']='ambiguous_device_link'
    with pytest.raises(ValueError):list(iter_authorized(io.BytesIO(),mapping,roles,allowed,lambda _:True))


def test_cap_stops_before_tail_with_quarantine_enabled():
    mapping,roles,allowed=fixture(); prefix=record('shared',poison=True)+record('device-yes')
    stream=io.BytesIO(prefix+b'bad-tail')
    assert len(list(iter_authorized(stream,mapping,roles,allowed,lambda _:True,max_per_user=1,
                                    quarantine_device_ids=[canonical(('string','shared'))])))==1
    assert stream.tell()==len(prefix)


@pytest.mark.parametrize('owners',[[], ['bad'], [canonical(('string','u'))]*2])
def test_invalid_audit_mapping_rejected(owners):
    with pytest.raises(ValueError):assign_quarantined({canonical(('string','d')):owners})
