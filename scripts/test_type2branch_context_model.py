import numpy as np
import pytest
from type2branch_context_model import context_hashes,MeanContextModel
from type2branch_synthesis_cleanup import INVALID_TIMING


def test_sentinel_and_partition_hashes():
    h=list(context_hashes([65,66,67],[2]))
    assert h[0]==[(0,65<<56),(1,(65<<56)|255)]
    assert h[1][-1]==(2,(66<<56)|0xff41)
    assert h[2]==[(0,67<<56),(1,(67<<56)|255)]
    assert max(o for o,_ in list(context_hashes(range(20)))[-1])==7


def test_minimum_count_sentinel_and_truncation():
    m=MeanContextModel()
    for i in range(9):m.feed(np.array([[65,100+i,INVALID_TIMING]]),[])
    assert np.isnan(m.predict([65])[0]).all()
    m.feed(np.array([[65,109,INVALID_TIMING]]),[])
    values,orders=m.predict([65])
    assert values[0,0]==104 and orders[0,0]==1
    assert np.isnan(values[0,1]) and orders[0,1]==-1


def test_longest_context_backoff_and_zero_hash_lookup():
    m=MeanContextModel()
    for _ in range(10):m.feed(np.array([[65,10,INVALID_TIMING],[66,20,100]]),[])
    for _ in range(9):m.feed(np.array([[67,30,INVALID_TIMING],[66,40,200]]),[])
    values,orders=m.predict([67,66])
    assert orders[1,0]==0 and values[1,0]==29
    values,orders=m.predict([65,66])
    assert orders[1,0]==2 and values[1,0]==20
    for _ in range(10):m.feed(np.array([[0,99,INVALID_TIMING]]),[])
    # At a nonmatching predecessor, generic key0 hash0 is deliberately skipped.
    assert np.isnan(m.predict([66,0])[0][1,0])


def test_statistics_against_direct_arithmetic():
    m=MeanContextModel()
    values=[11,25,8,1500,0,42,17,18,29,60]
    for v in values:m.feed(np.array([[65,v,INVALID_TIMING]]),[])
    count,mean,square=m.models[(0,0,65<<56)]
    assert count==10
    np.testing.assert_allclose([mean,square],[np.mean(values),np.mean(np.square(values))],rtol=1e-14)


@pytest.mark.parametrize('offsets',[[1,1],[2,1],[-1],[3],[.5]])
def test_invalid_partition_lists_rejected(offsets):
    with pytest.raises(ValueError):list(context_hashes([65,66,67],offsets))
