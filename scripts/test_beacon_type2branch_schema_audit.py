import pytest
from beacon_type2branch_schema_audit import milliseconds,audit_rows


def test_decimal_ms_without_float_rounding():
    assert milliseconds('186.415')==186415
    assert milliseconds('0.001')==1
    for value in ['.0001','NaN','Infinity','bad']:
        with pytest.raises(ValueError):milliseconds(value)


def test_release_order_and_exact_window_counts():
    rows=[{'Elapsed Start Time':f'{i:.3f}','Elapsed Release Time':f'{i+.1:.3f}',
           'Duration':'.100','Key':'a'} for i in range(30)]
    result=audit_rows(rows[::-1],[0.])
    assert result['release_order_onset_decreases']==29
    assert result['windows']==[{'start':0.,'real_events':30,'hypothetical_anchor':25}]
    assert result['median_duration_disagreement_ms']==0
    assert result['unknown_codes']==0


def test_whole_hold_must_fit_boundary():
    rows=[{'Elapsed Start Time':str(i),'Elapsed Release Time':str(i+1),'Duration':'1','Key':'a'} for i in range(30)]
    assert audit_rows(rows,[0.])['windows'][0]['real_events']==29
    rows[0]['Elapsed Release Time']='-1'
    with pytest.raises(ValueError,match='Negative hold'):audit_rows(rows,[0.])
