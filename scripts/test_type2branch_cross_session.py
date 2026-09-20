import numpy as np
import pytest
from type2branch_cross_session import cross_session_scores


def data():
    return [np.repeat([0.,10.],5)[:,None],np.repeat(['a','b'],5),np.tile(np.arange(15,20),2),
            np.repeat([1.,11.],10)[:,None],np.repeat(['a','b'],10),np.tile(np.arange(10),2),['a','b']]


def test_fixed_gallery_and_probe_membership():
    result=cross_session_scores(*data())
    assert result['scores'].shape==(20,2)
    np.testing.assert_array_equal(result['genuine'],np.full(20,-1.))
    np.testing.assert_array_equal(result['scores'][0],[-1.,-9.])


@pytest.mark.parametrize('fault',['gallery_window','probe_duplicate','identity','missing_cohort'])
def test_invalid_allocation(fault):
    args=data()
    if fault=='gallery_window':args[2][0]=0
    if fault=='probe_duplicate':args[5][1]=0
    if fault=='identity':args[4][0]='c'
    if fault=='missing_cohort':args[6]=['a','b','c']
    with pytest.raises(ValueError):cross_session_scores(*args)
