import pytest
import numpy as np
from type2branch_keyrecs_adapter import adapt, parse_event, milliseconds


def test_incoming_shift_negative_clamp_and_terminal_hold():
    rows=['a,a,.1,.25,.35,.15,.25,']*98
    rows+=['a,b,.1,-.05,.05,-.15,-.05,','b,,.1,,,,,']
    raw,base,info=adapt(rows)
    assert raw.shape==(1,100,3)
    assert raw[0,0,2]==0 and raw[0,1,2]==250 and raw[0,99,2]==-50
    assert base[0,99,2]==0 and base[0,99,1]==.1
    assert info['terminal_events']==1


def test_unknown_quote_key_preserves_timing():
    code,ht,dd,unknown,terminal=parse_event('"a,b,.1,.25,.35,.15,.25,',last=False)
    assert (code,ht,dd,unknown,terminal)==(0,100,250,True,False)


def test_missing_interior_and_real_submillisecond_values_rejected():
    with pytest.raises(ValueError):parse_event('a,,.1,,,,,',last=False)
    with pytest.raises(ValueError):milliseconds('.1000005')
    assert milliseconds('.10000000000000001')==100


def test_window_start_resets_and_long_timings_clip():
    rows=['a,a,31,40,71,9,40,']*200
    raw,base,_=adapt(rows)
    np.testing.assert_array_equal(raw[:,0,2],[0,40000])
    np.testing.assert_array_equal(base[:,0,2],[0,0])
    assert base[:,:,1:].max()==30
