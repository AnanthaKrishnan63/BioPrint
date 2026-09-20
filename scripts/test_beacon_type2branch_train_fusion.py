import copy
import numpy as np
import pytest
from beacon_type2branch_train_fusion import attach_embeddings, pairs


def fixture():
    data = {s: {role: {'windows': [{'start': float(i * 30)} for i in range(n)]}
                for role, n in [('enrollment', 6), ('probe', 1)]} for s in ['P002', 'P003']}
    keys = [(s, r, w['start']) for s, d in data.items() for r, v in d.items() for w in v['windows']]
    arrays = {'embeddings': np.array([np.full(256, i) for i in range(len(keys))], dtype=float),
              'subject': np.array([k[0] for k in keys]), 'role': np.array([k[1] for k in keys]),
              'start': np.array([k[2] for k in keys]), 'true_length': np.full(len(keys), 5)}
    return data, arrays, {k: 5 for k in keys}


def old_pairs(data):
    groups = [(o, a, str(w['start'])) for o in data for a in data for w in data[a]['probe']['windows']]
    return np.arange(len(groups) * 34).reshape(-1, 34), np.array([int(o == a) for o, a, _ in groups]), np.array(groups)


def test_exact_attachment_permits_metadata_reordering_without_mutation():
    data, arrays, lengths = fixture()
    before = copy.deepcopy(data)
    attached = attach_embeddings(data, {k: v[::-1] for k, v in arrays.items()}, lengths)
    assert data == before
    assert attached['P002']['probe']['windows'][0]['type2branch_embedding'] == [6.] * 256
    assert attached['P003']['enrollment']['windows'][0]['type2branch_true_length'] == 5


def test_pair_extension_preserves_old_columns_and_uses_first_five_gallery():
    data, arrays, lengths = fixture()
    data = attach_embeddings(data, arrays, lengths)
    x, y, groups = pairs(data, old_pairs)
    previous, py, pg = old_pairs(data)
    np.testing.assert_array_equal(x[:, :34], previous)
    np.testing.assert_array_equal(y, py)
    np.testing.assert_array_equal(groups, pg)
    # sqrt256=16; first gallery0..4 excludes sixth value5.
    assert x[0, 34] == np.mean(np.abs(np.arange(5) - 6)) * 16
    assert x[1, 34] == np.mean(np.abs(np.arange(5) - 13)) * 16


@pytest.mark.parametrize('defect', ['duplicate', 'length', 'subject', 'count', 'nonfinite', 'audit', 'reattach'])
def test_attachment_rejects_mismatch(defect):
    data, arrays, lengths = fixture()
    if defect == 'duplicate': arrays['start'][1] = arrays['start'][0]
    elif defect == 'length': arrays['true_length'][0] = 6
    elif defect == 'subject': arrays['subject'][0] = 'P004'
    elif defect == 'count': arrays['embeddings'] = arrays['embeddings'][:-1]
    elif defect == 'nonfinite': arrays['embeddings'][0, 0] = np.nan
    elif defect == 'audit': lengths.pop(next(iter(lengths)))
    elif defect == 'reattach': data = attach_embeddings(data, arrays, lengths)
    with pytest.raises(ValueError): attach_embeddings(data, arrays, lengths)


def test_claim_order_changes_rejected():
    data, arrays, lengths = fixture()
    data = attach_embeddings(data, arrays, lengths)
    def reversed_pairs(records):
        return tuple(v[::-1] for v in old_pairs(records))
    with pytest.raises(ValueError, match='ordering'):
        pairs(data, reversed_pairs)
