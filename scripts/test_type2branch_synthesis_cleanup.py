import numpy as np
from type2branch_synthesis_cleanup import cleanup, INVALID_TIMING


def test_exact_boundary_negative_and_partition_order():
    raw=np.array([[65,-1,-1],[66,1500,1500],[67,1501,1501],[68,0,-1]])
    before=raw.copy()
    cleaned,partitions=cleanup(raw)
    np.testing.assert_array_equal(partitions,[2])
    np.testing.assert_array_equal(cleaned,[[65,1500,INVALID_TIMING],[66,1500,1500],
                                          [67,1500,INVALID_TIMING],[68,0,1500]])
    np.testing.assert_array_equal(raw,before)


def test_first_flight_always_invalid_even_if_positive():
    cleaned,partitions=cleanup([[0,100,2000]])
    np.testing.assert_array_equal(cleaned,[[0,100,INVALID_TIMING]])
    np.testing.assert_array_equal(partitions,[0])


def test_negative_flight_not_a_pause_partition():
    cleaned,partitions=cleanup([[65,100,0],[66,100,INVALID_TIMING]])
    assert not len(partitions)
    assert cleaned[1,2]==1500
