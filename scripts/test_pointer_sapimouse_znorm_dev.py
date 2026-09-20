from copy import deepcopy
import pytest
import pointer_sapimouse_znorm_dev as runner


def test_record_references_and_thresholds_cannot_be_replaced(monkeypatch):
    background = [{'split': 'train', 'role': 'representation', 'path': 'x/bg/x_3min.csv'}]
    files = []
    for n in range(24):
        files.extend([{'split': 'train', 'role': 'dev_support_enrollment', 'path': f'x/user{n}/x_3min.csv'},
                      {'split': 'dev', 'role': 'dev_probe', 'path': f'x/user{n}/x_1min.csv'}])
    monkeypatch.setattr(runner, 'training_entries', lambda _: {'background': background})
    plan = {'background_entries': background, 'dev_entries': [e for e in files if e['split'] == 'dev'],
            'subjects': sorted(f'user{n}' for n in range(24)), 'thresholds': {'frozen': 1.}, 'selected': 'cosine@5'}
    train = {'thresholds': {'frozen': 1.}}
    runner.validate_records(plan, {'files': files}, train)
    for field, value in [('background_entries', []), ('dev_entries', [{'split': 'test_sealed'}]),
                         ('thresholds', {'frozen': 0.}), ('subjects', [])]:
        changed = deepcopy(plan); changed[field] = value
        with pytest.raises(ValueError): runner.validate_records(changed, {'files': files}, train)
