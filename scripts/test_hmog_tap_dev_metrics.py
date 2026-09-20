import numpy as np
import pytest
from hmog_tap_dev_metrics import METHODS, evaluate, summarize

ACCOUNTS = ['a', 'b', 'c', 'd']
THRESHOLDS = {'0.001': .1, '0.01': .5, '0.05': 1.}


def fixture():
    subjects = np.array(['a', 'b', 'c', 'd', 'a', 'b'])
    scores = np.full((6, 4), 3.)
    for row, subject in enumerate(subjects): scores[row, ACCOUNTS.index(subject)] = .2
    return subjects, scores, np.array([False] * 4 + [True] * 2)


def test_missing_probe_accounts_keep_impostor_claims_and_null_genuine_metrics():
    subjects, scores, probe = fixture()
    result = evaluate(scores, np.isfinite(scores), subjects, ACCOUNTS, probe, THRESHOLDS, 10)
    c = result['per_account']['c']
    assert c['eer'] is None
    assert c['rates']['0.01']['frr'] is None
    assert c['rates']['0.01']['impostor_count'] == 2
    assert result['base_window_rates']['0.01']['genuine_coverage'] == 1
    assert result['known_inspected_candidate_rates']['0.01']['genuine_coverage'] == .2
    assert result['known_inspected_candidate_rates']['0.01']['genuine_not_accepted_rate'] == .8


def test_missing_observation_keeps_original_opportunity():
    subjects, scores, probe = fixture()
    scores[-1] = np.nan
    r = evaluate(scores, np.isfinite(scores), subjects, ACCOUNTS, probe, THRESHOLDS, 10)
    assert r['base_window_rates']['0.01']['missing_genuine'] == 1
    assert r['base_window_rates']['0.01']['missing_impostor'] == 3
    assert r['base_window_rates']['0.01']['genuine_not_accepted_rate'] == .5


def test_complete_observed_comparison_does_not_erase_full_cohort_failure():
    subjects, scores, probe = fixture()
    result = summarize({m: scores for m in METHODS}, {m: np.isfinite(scores) for m in METHODS},
                       subjects, ACCOUNTS, probe, {m: THRESHOLDS for m in METHODS}, 10)
    assert result['status'] == 'infeasible'
    assert result['observed_window_comparison_completed']
    assert result['missing_genuine_probe_accounts'] == ['c', 'd']
    assert result['selection_outcome'] == 'no_useful_discrimination'
    assert 'selected_method' not in result


def test_candidate_denominator_cannot_be_smaller_than_original_probes():
    subjects, scores, probe = fixture()
    with pytest.raises(ValueError):
        evaluate(scores, np.isfinite(scores), subjects, ACCOUNTS, probe, THRESHOLDS, 1)
