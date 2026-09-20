import copy
import json
from brainrun_train_numeric_audit import Numeric, Profile, POINT_FIELDS


def test_numeric_invalid_values_counted_without_serializing_them():
    n=Numeric()
    for value in [0,3.,-1,1.5,True,{},float('nan'),float('inf')]:n.add(value)
    n.add(None,present=False)
    report=n.report()
    assert report['total']==9 and report['finite']==4 and report['integer']==3
    assert report['nonnegative']==3 and report['nonnegative_integer']==2
    assert report['invalid_type']==2 and report['nonfinite']==2 and report['missing']==1
    assert report['minimum']==-1 and report['maximum']==3
    json.dumps(report,allow_nan=False)


def test_game_durations_and_correctness_audit_no_exclusions():
    p=Profile('games')
    p.add('u',{'t_start':100,'t_stop':50,'corrects_number':-1,'wrongs_number':1.5,'game_type':'Mathisis','stage':2})
    p.add('u',{'t_start':'bad','t_stop':float('nan'),'corrects_number':True,'game_type':{},'stage':[]})
    p.add('v',['not','document'])
    report=p.report()
    assert report['counts']['records']==3 and report['counts']['invalid_document_shape']==1
    assert report['counts']['negative_duration']==1
    assert report['numeric']['duration']['minimum']==-50
    assert report['numeric']['duration']['missing']==1
    assert report['numeric']['corrects_number']['nonnegative_integer']==0
    assert report['numeric']['wrongs_number']['nonnegative_integer']==0
    assert report['authorized_document_counts']=={'u':2,'v':1}
    json.dumps(report,allow_nan=False)


def test_gesture_points_empty_arrays_sessions_and_unexpected_timestamps():
    p=Profile('gestures');point={name:1. for name in POINT_FIELDS};point.update(timestamp=123,other='not serialized')
    doc={'t_start':1,'t_stop':1,'type':'tap','screen':'Mathisis','session_id':'a','data':[point]}
    before=copy.deepcopy(doc);p.add('u',doc)
    p.add('u',{**doc,'data':[]})
    p.add('u',{**doc,'session_id':'b','data':[None,{'dx':True,'dy':float('nan'),3:'bad key'}]})
    p.add('v',{**doc,'session_id':{},'data':{}})
    report=p.report()
    assert doc==before
    assert report['session_counts_per_user']=={'u':2}
    assert report['counts']['empty_data_arrays']==1 and report['counts']['invalid_point_shape']==1
    assert report['counts']['invalid_data_shape']==1 and report['counts']['invalid_session_id']==1
    assert report['point_fields']['dx']['invalid_type']==1
    assert report['point_fields']['dy']['nonfinite']==1
    assert report['point_timestamp_fields']=={'timestamp':1}
    assert report['unexpected_point_fields']=={'timestamp':1,'other':1}
    assert 'not serialized' not in json.dumps(report,allow_nan=False)
