from io import StringIO
import pytest
from type2branch_keyrecs_eligibility import count_train_prefixes, cardinalities


def test_opaque_measurements_never_parsed_and_split_roles_preserved():
    lines=StringIO('participant,session,values\np001,1,not,numeric\np001,2,DEV\np999,1,SEALED\np001,1,"broken\n')
    assert count_train_prefixes(lines, {'p001'}) == {'p001': 2}


def test_fifteen_sequence_boundaries_and_no_overlap():
    assert cardinalities(2999)['fit_windows'] == 14
    assert cardinalities(3000)['fit_windows'] == 15
    assert cardinalities(6000)['selection_windows'] == 15
    for count in [0, 99, 100, 2999, 6000]:
        r=cardinalities(count)
        assert r['fit_windows']+r['selection_windows']+r['calibration_windows']==r['full_100_row_windows']
        assert r['full_100_row_windows']*100+r['tail_rows']==count


def test_wrong_header_fails_before_rows():
    with pytest.raises(ValueError):
        count_train_prefixes(iter(['wrong,header,x\n']), {'p001'})
