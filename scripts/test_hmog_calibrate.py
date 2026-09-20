"""Generated metadata/embeddings only; no participant arrays are read."""
import io
import json
import numpy as np
import pytest
from unittest.mock import patch
import hmog_calibrate as m


def fixture():
    subjects=np.array(['663153','962159','663153','962159'])
    roles=np.array(['train_enrollment']*2+['train_calibration']*2)
    sessions=np.array([1,1,9,9])
    a=np.zeros((4,64));a[:,0]=[0,10,1,9]
    return subjects,roles,sessions,{b:a.copy() for b in m.BRANCHES}


def test_scores_profiles_thresholds_same_encoder_branches():
    s,r,_,emb=fixture();metrics,arrays=m.calibrate_embeddings(emb,s,r)
    for b in m.BRANCHES:
        np.testing.assert_array_equal(arrays[b+'_distances'],[[1,9],[9,1]])
        assert metrics[b]['pooled_eer_diagnostic']==0
        for op in metrics[b]['operating_points'].values():
            assert op['threshold']==np.nextafter(9.,-np.inf)
            assert op['far']==0 and op['frr']==0
            assert len(op['per_account'])==2


def test_ties_do_not_partially_accept_or_hide_frr():
    s,r,_,emb=fixture()
    for b in emb:emb[b][:]=0
    metrics,_=m.calibrate_embeddings(emb,s,r)
    for b in m.BRANCHES:
        assert metrics[b]['pooled_eer_diagnostic']==.5
        for op in metrics[b]['operating_points'].values():
            assert op['threshold']<0 and op['far']==0 and op['frr']==1


def test_missing_account_infeasible():
    s,r,_,emb=fixture();mask=np.arange(4)!=3
    with pytest.raises(ValueError,match='Both calibration'):m.calibrate_embeddings({b:a[mask] for b,a in emb.items()},s[mask],r[mask])


@pytest.mark.parametrize('subject,session,role',[('556357',9,'dev_probe'),('180679',9,'train_selection'),('717868',9,'train_fit'),('663153',17,'sealed_test'),('663153',1,'train_calibration')])
def test_metadata_rejects_before_feature_copy(subject,session,role):
    class Poison:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            if key in ('key','imu'):raise AssertionError('FEATURES ACCESSED')
            return {'subject':np.array([subject]),'session':np.array([session]),'role':np.array([role])}[key]
    with patch.object(m.np,'load',return_value=Poison()):
        with pytest.raises(PermissionError):m.load_calibration_arrays('not-opened')


def coverage_report():
    subjects={s:{} for s in m.CALIBRATION_IDS};excluded={s:{} for s in m.CALIBRATION_IDS}
    for s in m.CALIBRATION_IDS:
        for i in range(1,17):
            if i in (1,9):subjects[s][str(i)]={'candidate_windows':2,'eligible_windows':1}
            else:excluded[s][str(i)]={'reason':'empty_keypress','basis':'archive_metadata','activity_contents_inspected':False,'candidate_windows':None,'eligible_windows':None}
    return {'subjects':subjects,'excluded_sessions':excluded}


def test_coverage_preserves_unknown_metadata_denominators():
    s,_,sessions,_=fixture();report=coverage_report();result=m.coverage_from_report(report,s,sessions)
    assert result['663153']['train_calibration']['total_candidate_windows'] is None
    assert result['663153']['train_calibration']['known_excluded_windows']==1
    assert len(result['663153']['train_calibration']['uninspected_sessions'])==7
    report['subjects']['663153']['9']['eligible_windows']=2
    with pytest.raises(ValueError,match='count mismatch'):m.coverage_from_report(report,s,sessions)


def test_selection_provenance_and_earliest_tie():
    report={'status':'complete','selection_only':True,'calibration_performed':False,'dev_accessed':False,'plan':{'model_protocol_sha256':'p'},'selected_epoch':1,'selected_checkpoint_sha256':'a'*64,'history':[{'epoch':i,'checkpoint_sha256':'a'*64,'metrics':{'joint':{'pooled_eer':.5}}} for i in range(1,21)]}
    assert m.validate_selection(report,'p','h','h')=='a'*64
    with pytest.raises(ValueError,match='checksum'):m.validate_selection(report,'p','bad','h')
    report['selected_epoch']=2
    with pytest.raises(ValueError,match='objective'):m.validate_selection(report,'p','h','h')


def test_nonfinite_embeddings_rejected():
    s,r,_,emb=fixture();emb['joint'][0,0]=np.nan
    with pytest.raises(ValueError):m.calibrate_embeddings(emb,s,r)
