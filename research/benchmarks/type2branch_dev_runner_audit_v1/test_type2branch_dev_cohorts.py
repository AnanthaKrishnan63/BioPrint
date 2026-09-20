import numpy as np
from type2branch_cross_session import cross_session_scores
from type2branch_evaluate_dev import cohort_results


def test_cohorts_use_declared_claim_denominators():
    ids=np.array(['a','b','c','d'])
    gallery=np.repeat(np.arange(4)*100.,5)[:,None]
    probes=np.repeat(np.arange(4)*100.+1,10)[:,None]
    scored=cross_session_scores(gallery,np.repeat(ids,5),np.tile(np.arange(15,20),4),
        probes,np.repeat(ids,10),np.tile(np.arange(10),4),ids)
    result=cohort_results(scored,{'first':['a','b'],'second':['c','d']},-2.)
    for role in result.values():
        local=role['within_cohort_claims']['pooled']
        full=role['full_account_claims_grouped_by_probe_identity']
        assert local['genuine_n']==full['genuine_n']==20
        assert local['impostor_n']==20 and full['impostor_n']==60
        assert local['far']==local['frr']==full['far']==full['frr']==0
