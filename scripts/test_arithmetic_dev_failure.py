from arithmetic_dev_failure import check_block


def test_fifty_trial_block_is_reported_not_truncated():
    data = {'imageT': list(range(50)), 'keyT': [x + .5 for x in range(50)],
            'CorrectAns': [1] * 50}
    result = check_block(data)
    assert result['status'] == 'invalid'
    assert result['shapes'] == {key: [50] for key in data}
    assert data['CorrectAns'] == [1] * 50
