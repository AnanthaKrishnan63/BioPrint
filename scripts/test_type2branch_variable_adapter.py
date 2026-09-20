import numpy as np
import pytest
from type2branch_variable_adapter import adapt_variable, synthesize_window
from type2branch_keyrecs_adapter import adapt
from type2branch_input_audit import reference_prepare
from type2branch_random import first_synthesis_thread


def rows(count):
    return ['a,a,.1,.25,.35,.15,.25,']*(count-1)+['a,,.1,,,,,']


@pytest.mark.parametrize('count',[1,25,50,75,99,100,101,199,225])
def test_author_base_padding_and_lengths(count):
    windows,audit=adapt_variable(rows(count),max_windows=10)
    assert sum(audit['true_lengths'])==count and audit['terminal_events']==1
    for window in windows:
        wire=window['raw_ms'];press=np.r_[0,np.cumsum(wire[1:,2])]
        events=np.column_stack([press,press+wire[:,1],wire[:,0]])
        expected=reference_prepare()(events,1000.)
        actual=np.pad(window['base'],((0,100-len(wire)),(0,0)))
        np.testing.assert_array_equal(actual,expected)


def test_full_windows_match_frozen_adapter():
    source=rows(225);old_raw,old_base,_=adapt(source)
    windows,_=adapt_variable(source,max_windows=10)
    for i in range(2):
        np.testing.assert_array_equal(windows[i]['raw_ms'],old_raw[i])
        np.testing.assert_array_equal(windows[i]['base'],old_base[i])
    assert windows[2]['true_length']==25


def test_cut_not_terminal_and_unparsed_tail():
    source=['a,a,.1,.25,.35,.15,.25,']*100+['not parseable']
    windows,audit=adapt_variable(source,max_windows=1)
    assert audit['unparsed_rows']==1 and not audit['actual_session_end']
    source[99]='a,,.1,,,,,'
    with pytest.raises(ValueError):adapt_variable(source,max_windows=1)
    with pytest.raises(ValueError):adapt_variable(rows(25),max_windows=1,session_complete=False)


def test_padding_never_reaches_population_or_random_fallback():
    class Population:
        def predict(self,keys):
            assert len(keys)==25
            return np.full((25,2),np.nan),np.full((25,2),-1,dtype=np.int16)
    class Counting:
        def __init__(self):self.calls=0
        def next_double(self):self.calls+=1;return .5
    window=adapt_variable(rows(25),max_windows=1)[0][0];rng=Counting()
    result=synthesize_window(window,Population(),rng)
    assert rng.calls==50 and result['true_length']==25
    np.testing.assert_array_equal(result['features'][25:],0)
    np.testing.assert_array_equal(result['context_orders'][25:],-2)
    assert not result['residual_valid'][25:].any()
    assert result['features'][0,4]==0 and not result['residual_valid'][0,1]


def test_true_zero_key_is_not_inferred_as_padding():
    class Population:
        def predict(self,keys):return np.full((len(keys),2),100.),np.zeros((len(keys),2),dtype=np.int16)
    window={'raw_ms':np.array([[0,0,0]]),'base':np.zeros((1,3)),'true_length':1}
    result=synthesize_window(window,Population(),first_synthesis_thread())
    assert result['true_length']==1 and result['residual_valid'][0,0]
    assert result['features'][0,3]==-.1
    assert not result['residual_valid'][1:].any()


def test_one_event_consumes_both_fallback_draws_before_cleanup():
    class Population:
        def predict(self,keys):
            assert len(keys)==1
            return np.full((1,2),np.nan),np.full((1,2),-1,dtype=np.int16)
    class Counting:
        def __init__(self):self.calls=0
        def next_double(self):self.calls+=1;return self.calls/10
    window=adapt_variable(rows(1),max_windows=1)[0][0]
    rng=Counting();result=synthesize_window(window,Population(),rng)
    assert rng.calls==2
    assert result['synthetic_real_ms'][0,1]==100
    np.testing.assert_array_equal(result['residual_valid'][0],[True,False])
    np.testing.assert_array_equal(result['context_orders'][0],[-1,-1])
    assert not result['residual_valid'][1:].any()


def test_full_synthesis_matches_existing_pipeline_and_rng_state():
    from type2branch_random import fill_average_fallback
    from type2branch_synthesis_cleanup import cleanup
    from type2branch_residual_features import residual_features
    class Population:
        def predict(self,keys):
            predictions=np.full((len(keys),2),120.)
            predictions[::3]=np.nan
            return predictions,np.where(np.isnan(predictions),-1,0)
    source=rows(100);original=list(source)
    raw,base,_=adapt(source);rng_old=first_synthesis_thread();rng_new=first_synthesis_thread()
    predicted,_=Population().predict(raw[0,:,0])
    clean,_=cleanup(np.column_stack((raw[0,:,0],fill_average_fallback(predicted,rng_old))))
    expected,valid=residual_features(base[0],clean)
    window=adapt_variable(source,max_windows=1)[0][0]
    raw_before=window['raw_ms'].copy();base_before=window['base'].copy()
    result=synthesize_window(window,Population(),rng_new)
    np.testing.assert_array_equal(result['features'],expected)
    np.testing.assert_array_equal(result['residual_valid'],valid)
    assert vars(rng_old)==vars(rng_new)
    np.testing.assert_array_equal(window['raw_ms'],raw_before)
    np.testing.assert_array_equal(window['base'],base_before)
    assert source==original


def test_cross_window_key_continuity_is_validated():
    source=rows(101);source[100]='b,,.1,,,,,'
    with pytest.raises(ValueError,match='Adjacent key mismatch'):
        adapt_variable(source,max_windows=2)


@pytest.mark.parametrize('limit',[0,21,True,1.5,None])
def test_invalid_limits_rejected(limit):
    with pytest.raises(ValueError):adapt_variable(rows(1),max_windows=limit)


def test_empty_capture_rejected():
    with pytest.raises(ValueError):adapt_variable([],max_windows=1)
