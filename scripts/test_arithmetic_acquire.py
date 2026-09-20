import pytest
from arithmetic_acquire import make_plan


def tree():
    return {'sha': 'example', 'tree': [{'path': f'{i}.zip', 'size': 10}
                                     for i in range(19)]}


def test_frozen_split_and_test_exclusion():
    first, second = make_plan(tree()), make_plan(tree())
    assert first['entries'] == second['entries']
    roles = [r['role'] for r in first['entries']]
    assert [roles.count(r) for r in ['train_fit', 'train_calibration', 'dev', 'test_sealed']] == [7, 4, 4, 4]
    assert first['planned_bytes'] == 150


def test_reject_incomplete_or_escaping_metadata():
    t = tree(); t['truncated'] = True
    with pytest.raises(ValueError):
        make_plan(t)
    t = tree(); t['tree'][0]['path'] = '../escape.zip'
    with pytest.raises(ValueError):
        make_plan(t)
