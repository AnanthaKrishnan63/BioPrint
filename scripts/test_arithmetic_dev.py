import pytest
from arithmetic_dev import dev_records


def test_dev_inventory_excludes_train_and_test_and_requires_all_accounts():
    entries = [{'path': f'{i}.zip', 'role': 'dev'} for i in range(4)]
    entries += [{'path': 'sealed.zip', 'role': 'test_sealed'}, {'path': 'fit.zip', 'role': 'train_fit'}]
    assert len(dev_records({'entries': entries})) == 4
    with pytest.raises(ValueError): dev_records({'entries': entries[1:]})
    entries[0]['path'] = '../0.zip'
    with pytest.raises(ValueError): dev_records({'entries': entries})
