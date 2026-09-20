import copy
import pytest
from type2branch_capture_api_replay import metric_difference


def test_subgroup_eer_difference_not_hidden_by_pooled_agreement():
    reference={'pooled':{'far':0.,'frr':.5},'by_length':{'25':{'margin_eer':.2}},'empty':{},'fraction':None}
    changed=copy.deepcopy(reference);changed['by_length']['25']['margin_eer']=.3
    assert metric_difference(reference,reference)==0
    assert metric_difference(changed,reference)==pytest.approx(.1)


@pytest.mark.parametrize('actual,expected',[
    ({'far':0.},{}),({'far':float('nan')},{'far':0.}),
    ({'fraction':0.},{'fraction':None}),({'counts':[1,2]},{'counts':[1]}),
    ({'identity':'different'},{'identity':'generated'}),({'count':True},{'count':1})])
def test_schema_missingness_and_nonfinite_rejected(actual,expected):
    with pytest.raises(ValueError):metric_difference(actual,expected)
