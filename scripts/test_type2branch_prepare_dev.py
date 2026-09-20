import numpy as np
import pytest
from type2branch_prepare_dev import select_dev_rows, require_allocation, adapt_prescribed_prefix


def test_sealed_and_other_session_payloads_stay_opaque():
    grouped=select_dev_rows(iter(['participant,session,key1,key2,rest\n',
        'active,1,gallery\n','active,2,probe\n','test,1,invalid payload\n',
        'test,2,invalid payload\n','active,3,invalid payload\n']),{'active'})
    assert dict(grouped)=={('active','1'):['gallery\n'],('active','2'):['probe\n']}


def test_exact_allocation_no_replacement():
    raw=np.arange(20)[:,None];base=raw/100
    a,b=require_allocation(raw,base,list(range(15,20)))
    np.testing.assert_array_equal(a[:,0],np.arange(15,20))
    np.testing.assert_array_equal(b[:,0],np.arange(15,20)/100)
    for windows in [[],[-1],[20],[1,1]]:
        with pytest.raises(ValueError):require_allocation(raw,base,windows)


def test_unselected_tail_is_not_parsed_and_cut_is_not_terminal():
    rows=['a,a,.1,.25,.35,.15,.25,']*100+['invalid unselected tail']
    raw,base,audit=adapt_prescribed_prefix(rows,[0])
    assert raw.shape==base.shape==(1,100,3) and audit['events']==100
    rows[99]='a,,.1,,,,,'
    with pytest.raises(ValueError):adapt_prescribed_prefix(rows,[0])
    raw,base,audit=adapt_prescribed_prefix(rows[:100],[0])
    assert audit['terminal_events']==1
