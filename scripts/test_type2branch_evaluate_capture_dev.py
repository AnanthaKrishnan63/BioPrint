import json
import numpy as np
import pytest
import type2branch_evaluate_capture_dev as evaluator


def test_gallery_assembly_preserves_identity_and_window_order():
    identities=np.array(['a','b']);subjects=np.repeat(identities,5);windows=np.tile(np.arange(15,20),2)
    embeddings=np.arange(30).reshape(10,3)
    actual=evaluator.assemble_gallery(embeddings[::-1],subjects[::-1],windows[::-1],identities)
    np.testing.assert_array_equal(actual,embeddings.reshape(2,5,3))


@pytest.mark.parametrize('fault',['missing','duplicate','nan','foreign'])
def test_bad_gallery_fails(fault):
    identities=np.array(['a','b']);subjects=np.repeat(identities,5);windows=np.tile(np.arange(15,20),2)
    embeddings=np.ones((10,3))
    if fault=='missing':embeddings=embeddings[:-1];subjects=subjects[:-1];windows=windows[:-1]
    if fault=='duplicate':windows[1]=windows[0]
    if fault=='nan':embeddings[0,0]=np.nan
    if fault=='foreign':subjects[0]='c'
    with pytest.raises(ValueError):evaluator.assemble_gallery(embeddings,subjects,windows,identities)


def test_infeasible_preparation_rejected_before_arrays_and_output(tmp_path,monkeypatch):
    features=tmp_path/'research/benchmarks/type2branch_capture_dev_features_v1';features.mkdir(parents=True)
    (features/'report.json').write_text(json.dumps({'status':'infeasible_prescribed_cohort','failures':[{'subject':'generated'}]}))
    monkeypatch.setattr(evaluator,'ROOT',tmp_path);monkeypatch.setattr(evaluator,'OUT',tmp_path/'output')
    monkeypatch.setattr(evaluator.np,'load',lambda *a,**k:pytest.fail('Premature DEV array read'))
    with pytest.raises(ValueError,match='no subset scoring'):evaluator.main()
    assert not evaluator.OUT.exists()
