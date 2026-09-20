import numpy as np
import pytest
from type2branch_length_audit import truncate_features
from type2branch_input_audit import reference_prepare
from type2branch_random import fill_average_fallback
from type2branch_wire import validate_features


@pytest.mark.parametrize('length',[1,25,50,75,100])
def test_author_base_padding_matches_feature_truncation(length):
    press=np.arange(length)*200
    events=np.column_stack([press,press+100,np.full(length,65)])
    prepared=reference_prepare()(events,1000.)
    full=np.ones((1,100,5));full[0,:length,:3]=prepared[:length]
    original=full.copy();actual=truncate_features(full,length)
    np.testing.assert_array_equal(actual[0,:,:3],prepared)
    np.testing.assert_array_equal(actual[0,length:,3:],0)
    np.testing.assert_array_equal(full,original)
    validate_features(actual)


@pytest.mark.parametrize('length',[0,101,True,1.5])
def test_invalid_lengths(length):
    with pytest.raises(ValueError):truncate_features(np.zeros((1,100,5)),length)


def test_short_raw_synthesis_is_not_full_feature_truncation():
    class Draws:
        def __init__(self):self.index=0
        def next_double(self):
            self.index+=1
            return self.index/10
    full=fill_average_fallback(np.full((2,2),np.nan),Draws())
    short=fill_average_fallback(np.full((1,2),np.nan),Draws())
    assert full[0,0]==short[0,0]==100
    assert full[0,1]==300 and short[0,1]==200
