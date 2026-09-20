from type2branch_order_audit import summarize


def test_terminal_record_and_mixed_press_release_order():
    result=summarize(['a,b,.2,-.1,.3,-.3,.1,', 'b,,.4,,,,,'])
    assert result['counts']['dd_negative_uu_nonnegative']==1
    assert result['incomplete_records']==[{'index':1,'is_first':False,'is_last':True,
                                          'missing_fields':4,'empty_keys':1}]


def test_nonterminal_missing_record_not_hidden():
    result=summarize(['a,,.1,,,,,','a,b,.1,.25,.35,.15,.25,'])
    assert result['incomplete_records'][0]['is_first']
    assert not result['incomplete_records'][0]['is_last']
