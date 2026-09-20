import numpy as np
import pytest
from hmog_calibrate_fallback import partition,selected_feature_rows,score,FIT_IDS,BRANCHES

def metadata():
    s=np.repeat(np.array(sorted(FIT_IDS)),3);u=np.tile([2,7,9],4);r=np.tile(['train_enrollment','train_enrollment','train_fit'],4)
    return s,u,r

def test_disjoint_numeric_sessions_and_optimizer_exclusion():
    s,u,r=metadata();selected,labels,audit=partition(s,u,r)
    assert len(selected)==8 and np.all(u[selected]<=8)
    assert labels.tolist()==['gallery','calibration']*4
    for a in audit.values():assert a['gallery_sessions']==[2] and a['calibration_sessions']==[7]
    with pytest.raises(ValueError):partition(s[u!=7],u[u!=7],r[u!=7])
    s[0]='556357'
    with pytest.raises(PermissionError):partition(s,u,r)

def test_excluded_row_values_not_decoded_or_validated(tmp_path):
    data=np.ones((3,50,10),dtype=np.float32);data[1]=np.nan
    p=tmp_path/'synthetic.npz';np.savez_compressed(p,key=data)
    got=selected_feature_rows(p,'key',[0,2],3,(50,10))
    np.testing.assert_array_equal(got,np.ones((2,50,10)))
    with pytest.raises(ValueError,match='Nonfinite'):selected_feature_rows(p,'key',[1],3,(50,10))

def test_four_account_tied_scores_and_profiles():
    s,u,r=metadata();selected,labels,audit=partition(s,u,r);s=s[selected]
    em={b:np.zeros((8,64)) for b in BRANCHES};metrics,a=score(em,s,labels)
    for b in BRANCHES:
        assert metrics[b]['genuine_count']==4 and metrics[b]['impostor_count']==12
        assert metrics[b]['pooled_eer_diagnostic']==.5
        assert a[b+'_gallery'].shape==(4,64)
        for op in metrics[b]['operating_points'].values():assert op['far']==0 and op['frr']==1 and op['threshold']<0
