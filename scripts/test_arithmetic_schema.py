import copy
import pytest
from arithmetic_schema import validate_scope


def fixture():
    entries = [{'archive': f'{i}.zip', 'subject': str(i), 'members':
                [f'Cal_{i}_L{level}T{trial}.mat' for level in 'lmh' for trial in range(2, 7)]}
               for i in range(7)]
    return {'permitted_role': 'train_fit', 'entries': entries}, {
        'entries': [{'path': f'{i}.zip', 'role': 'train_fit' if i < 7 else 'dev'} for i in range(8)]}


def test_scope_rejects_dev_practice_and_other_modality():
    schema, acquisition = fixture()
    validate_scope(schema, acquisition)
    for member in ['Cal_0_LlT1.mat', 'Lin_0_LlT2.mat', '../Cal_0_LlT2.mat']:
        bad = copy.deepcopy(schema); bad['entries'][0]['members'][0] = member
        with pytest.raises(ValueError):
            validate_scope(bad, acquisition)
    bad = copy.deepcopy(schema); bad['entries'][0]['archive'] = '7.zip'
    with pytest.raises(ValueError):
        validate_scope(bad, acquisition)


def test_scope_rejects_duplicate_archive():
    schema, acquisition = fixture()
    schema['entries'].append(schema['entries'][0])
    with pytest.raises(ValueError):
        validate_scope(schema, acquisition)
