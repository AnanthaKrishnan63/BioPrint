from type2branch_prepare_inner_train import select_rows


def test_role_and_session_guard_leaves_unselected_payload_opaque():
    lines = iter(['participant,session,key1,key2,rest\n',
                  'selected,1,accepted payload\n',
                  'selected,2,not parseable timings\n',
                  'sealed,1,not parseable timings\n',
                  'fit,1,not parseable timings\n'])
    assert dict(select_rows(lines, {'selected'})) == {'selected':['accepted payload\n']}
