import pytest
from beacon_type2branch_prepare_dev import guard_role


@pytest.mark.parametrize('role,partition,ledger_role', [
    ('enrollment','train','personal_enrollment'), ('probe','dev','probe')])
def test_dev_support_is_training_without_global_fitting(role,partition,ledger_role):
    row={'split':'dev','role':role}
    entry={'cohort':'dev','storage_partition':'dev','record_partition':partition,
           'role':ledger_role,'global_model_fit_allowed':False}
    guard_role(row,entry)


@pytest.mark.parametrize('field,value', [('cohort','test'),('storage_partition','train'),
    ('record_partition','dev'),('role','probe'),('global_model_fit_allowed',True),
    ('global_model_fit_allowed',0)])
def test_enrollment_permissions_fail_closed(field,value):
    row={'split':'dev','role':'enrollment'}
    entry={'cohort':'dev','storage_partition':'dev','record_partition':'train',
           'role':'personal_enrollment','global_model_fit_allowed':False}
    entry[field]=value
    with pytest.raises(ValueError):guard_role(row,entry)


@pytest.mark.parametrize('split',['train','test'])
def test_other_observation_cohorts_rejected(split):
    with pytest.raises(ValueError):guard_role({'split':split,'role':'probe'}, {})
