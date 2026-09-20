from io import StringIO
from type2branch_continuity import fitting_rows, diagnose


def row(i):
    return ([str(i), str(i+1)], [.1,.25,.35,.15,.25])


def test_exact_chain_and_broken_key_boundary():
    rows=[row(i) for i in range(200)]
    assert diagnose(rows)['candidate_nonoverlapping_100row_windows']==2
    rows[99]=None
    r=diagnose(rows)
    assert r['candidate_nonoverlapping_100row_windows']==1
    assert r['counts']['malformed_rows']==1


def test_timing_failure_breaks_chain():
    rows=[row(i) for i in range(100)]
    rows[50]=(['50','51'], [.1,.25,.35,.16,.25])
    r=diagnose(rows)
    assert r['counts']['within_row_identity_failures']==1
    assert r['candidate_nonoverlapping_100row_windows']==0


def test_role_guard_before_measurement_decode():
    source=StringIO('participant,session,key1,key2,x\np001,1,a,b,.1,.25,.35,.15,.25,\np001,2,notparseable\np999,1,notparseable\n')
    rows=list(fitting_rows(source,{'p001'}))
    assert len(rows)==1 and rows[0][1] is not None
