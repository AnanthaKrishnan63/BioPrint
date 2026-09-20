import io
import pytest
from brainrun_record_identity_audit import IdentityAudit, canonical
from brainrun_bson_metadata import iter_metadata
from test_brainrun_guarded_stream import record


def typed(value):return ('string',value)
def key(value):return canonical(typed(value))
def audit():return IdentityAudit({key('d'):key('u')},[key('shared')],[key('u'),key('v')])


def test_metadata_accounting_and_unknown_user_sets():
    a=audit()
    for row in [{'device_id':typed('d'),'user_id':typed('u')},
                {'device_id':typed('d'),'user_id':typed('v')},
                {'device_id':typed('shared')}, {},
                {'device_id':typed('x'),'user_id':typed('u')},
                {'device_id':typed('x'),'user_id':typed('u')},
                {'device_id':typed('x'),'user_id':typed('outside')},
                {'device_id':typed('x')}]:a.observe(row)
    report=a.report(); counts=report['counts']
    assert counts['records']==8 and counts['known']==2 and counts['quarantine']==1
    assert counts['unknown']==4 and counts['missing_device']==1 and counts['conflicting_user']==1
    assert counts['explicit_user_present']==5 and counts['explicit_user_missing']==3
    assert counts['explicit_user_outside_frozen_users']==1
    assert report['unknown_devices'][key('x')]=={
        'records':4,'explicit_users':sorted([key('u'),key('outside')]),'missing_user_records':1}
    assert report['conflicting_device_users']=={key('d'):{key('v'):1}}


def test_poison_behavior_skipped_through_streaming_metadata_reader():
    a=audit()
    raw=record('d','u',poison=True)+record('unknown','outside',poison=True)
    for metadata in iter_metadata(io.BytesIO(raw),fields=['device_id','user_id']):a.observe(metadata)
    assert a.report()['counts']['records']==2


def test_nonidentity_fields_rejected():
    with pytest.raises(ValueError):audit().observe({'behavior':typed('opaque')})


def test_frozen_mapping_validation():
    with pytest.raises(ValueError):IdentityAudit({key('d'):key('u')},[key('d')],[key('u')])
    with pytest.raises(ValueError):IdentityAudit({key('d'):key('missing')},[],[key('u')])


def test_type_tags_are_distinct_for_unknown_device_associations():
    a=audit()
    a.observe({'device_id':('int32',12),'user_id':typed('u')})
    a.observe({'device_id':typed('12'),'user_id':typed('u')})
    assert a.report()['unknown_device_count']==2
