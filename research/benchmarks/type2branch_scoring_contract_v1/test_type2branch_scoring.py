import ast
from pathlib import Path
import numpy as np
import pytest
from type2branch_scoring import gallery_scores, chronological_scores, global_threshold, accept_scores


def test_score_against_actual_author_function():
    path=Path(__file__).resolve().parents[1]/'research/benchmarks/references/type2branch_evaluation/evaluate.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_score']
    assert len(nodes)==1
    scope={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    rng=np.random.default_rng(934)
    query=rng.normal(size=(35,8));gallery=rng.normal(size=(3,5,8))
    expected=np.array([[-scope['calculate_score'](g,q) for g in gallery] for q in query])
    np.testing.assert_allclose(gallery_scores(query,gallery),expected,rtol=1e-14,atol=1e-14)


def test_chronological_protocol_and_no_gallery_probe_overlap():
    subjects=np.repeat(['a','b'],15);windows=np.tile(np.arange(15),2)
    x=np.r_[np.arange(15),100+np.arange(15)][:,None].astype(float)
    order=np.arange(30)[::-1]
    result=chronological_scores(x[order],subjects[order],windows[order])
    assert result['scores'].shape==(20,2)
    assert len(result['genuine'])==len(result['impostor'])==20
    assert result['scores'][0,0]==-3  # query5 vs gallery0..4
    assert result['scores'][0,1]==-97 # query5 vs gallery100..104
    windows[1]=0
    with pytest.raises(ValueError):chronological_scores(x,subjects,windows)


@pytest.mark.parametrize('values,target', [([1,1,1],.01),([0,1,2,3],.25),([0,1,1,2],.5),([0],0)])
def test_threshold_ties_and_inclusive_decision(values,target):
    values=np.asarray(values,dtype=float)
    threshold=global_threshold(values,target)
    assert np.mean(values>=threshold)<=target
    # Including the next excluded tied score would exceed the allowed budget.
    lower=np.nextafter(threshold,-np.inf)
    assert np.mean(values>=lower)>target


@pytest.mark.parametrize('values,target', [([], .01),([np.nan],.01),([0],1),([0],-.1)])
def test_invalid_calibration(values,target):
    with pytest.raises(ValueError):global_threshold(values,target)


def test_float32_ties_preserve_threshold_and_2400_comparison_budget():
    tied=np.ones(2400,dtype=np.float32)
    assert accept_scores(tied,global_threshold(tied)).sum()==0
    increasing=np.arange(2400,dtype=np.float32)
    assert accept_scores(increasing,global_threshold(increasing)).sum()==24
    increasing[-25:]=2400
    assert accept_scores(increasing,global_threshold(increasing)).sum()==0
    threshold=1.0
    np.testing.assert_array_equal(accept_scores([np.nextafter(1.,0.),1.,2.],threshold),[False,True,True])
