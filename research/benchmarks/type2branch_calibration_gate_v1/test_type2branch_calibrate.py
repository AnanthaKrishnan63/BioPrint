import copy
import numpy as np
import pytest
from type2branch_calibrate import completed_checkpoint, metrics
from type2branch_scoring import chronological_scores


def fixture():
    plan={'epochs':3,'steps_per_epoch':100}
    records=[{'epoch':i,'updates':(i+1)*100,'selection_loss':v,'checkpoint':str(i)}
             for i,v in zip(range(-1,3),[1.,.8,.8,.9])]
    report={'status':'initial_budget_complete','optimizer_updates':300,'checkpoints':records,
            'best_checkpoint':records[1]}
    return report,plan


def test_completed_gate_and_earliest_tie():
    report,plan=fixture()
    assert completed_checkpoint(report,plan)['epoch']==0


@pytest.mark.parametrize('fault',['incomplete','updates','missing','choice','nan','checkpoint_updates'])
def test_gate_rejects_incomplete_or_changed_selection(fault):
    report,plan=fixture();report=copy.deepcopy(report)
    if fault=='incomplete':report['status']='running'
    if fault=='updates':report['optimizer_updates']=200
    if fault=='missing':report['checkpoints'].pop()
    if fault=='choice':report['best_checkpoint']=report['checkpoints'][2]
    if fault=='nan':report['checkpoints'][0]['selection_loss']=np.nan
    if fault=='checkpoint_updates':report['checkpoints'][1]['updates']=99
    with pytest.raises(ValueError):completed_checkpoint(report,plan)


def test_metrics_perfect_and_tied():
    subjects=np.repeat(['a','b'],15);windows=np.tile(np.arange(15),2)
    x=np.repeat([0.,10.],15)[:,None]
    result=metrics(chronological_scores(x,subjects,windows),-1.)
    assert result['pooled']['far']==result['pooled']['frr']==result['pooled']['discrete_eer']==0
    tied=metrics(chronological_scores(x*0,subjects,windows),0.)
    assert tied['pooled']['far']==1 and tied['pooled']['frr']==0
    assert tied['pooled']['discrete_eer']==.5
