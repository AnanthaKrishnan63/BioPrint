from type2branch_invalid_rows import classify


def test_zero_negative_and_fractional_milliseconds_separate():
    zero=classify('a,b,.1,0,.1,-.1,0,')
    negative=classify('a,b,.1,-.01,.1,-.11,0,')
    fractional=classify('a,b,.1000005,.25,.35,.15,.25,')
    assert zero['dd_zero_prior_decodable']==1 and zero['dd_negative_prior_decodable']==0
    assert negative['dd_negative_prior_decodable']==1 and negative['dd_zero_prior_decodable']==0
    assert fractional['submillisecond_timing_fields']==1


def test_missing_and_nonfinite_timing_fields_are_separate():
    r=classify('a,b,,nan,bad,.1,.2,')
    assert r['missing_timing_fields']==r['nonfinite_timing_fields']==r['nonnumeric_timing_fields']==1
    assert r['prior_undecodable_rows']==1


def test_bad_key_csv_does_not_hide_valid_timings():
    r=classify('"a,b,.1,.25,.35,.15,.25,')
    assert r['key_csv_error']==1 and r['complete_finite_timing_rows']==1
    assert r['prior_undecodable_rows']==1
