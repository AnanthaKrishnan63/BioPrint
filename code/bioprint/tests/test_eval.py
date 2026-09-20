"""Unit tests for the CMU harness (eval/cmu.py). OWNER: Agent E (eval).

Legacy protocol math is checked on synthetic fixtures only. Real test data stays sealed.
"""

from __future__ import annotations

import numpy as np
import pytest
import csv

from eval import cmu


@pytest.fixture()
def synthetic_csv(tmp_path):
    path = tmp_path / 'synthetic-keystrokes.csv'
    columns, _ = cmu.cmu_and_live_names()
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subject', 'sessionIndex', 'rep'] + columns)
        for user in range(3):
            for session in range(1, 9):
                for rep in range(1, 51):
                    writer.writerow([f's00{user+2}', session, rep] + [.1491 + user * .05] * len(columns))
    return path


def test_names_match_the_cmu_header_and_live_convention():
    cmu_names, live = cmu.cmu_and_live_names()
    assert len(cmu_names) == len(live) == 31
    assert cmu_names[:4] == ["H.period", "DD.period.t", "UD.period.t", "H.t"]
    assert live[:4] == ["H.Period#0", "DD.Period#0.KeyT#1", "UD.Period#0.KeyT#1", "H.KeyT#1"]
    assert live[-1] == "H.Enter#10"


def test_loader_shapes_and_units(synthetic_csv):
    ds = cmu.load(synthetic_csv, synthetic_only=True)
    assert len(ds.subjects) == 3 and len(ds.names) == 31
    s = ds.subjects["s002"]
    assert s.X.shape == (400, 31)
    assert list(s.session[:2]) == [1, 1] and list(s.rep[:2]) == [1, 2]
    assert set(s.session) == set(range(1, 9))
    # Synthetic milliseconds conversion, independent of actual CMU values.
    assert s.X[0, 0] == pytest.approx(149.1)
    holds = s.X[:, [j for j, n in enumerate(ds.names) if n.startswith("H.")]]
    assert 30 < np.median(holds) < 400  # human dwell in ms, not seconds


def test_without_return_drops_exactly_the_enter_columns(synthetic_csv):
    ds = cmu.load(synthetic_csv, synthetic_only=True).without_return()
    assert len(ds.names) == 28
    assert not any("Enter#" in n for n in ds.names)
    assert ds.subjects["s002"].X.shape == (400, 28)


def test_splits_are_the_documented_ones(synthetic_csv):
    s = cmu.load(synthetic_csv, synthetic_only=True).subjects["s002"]
    km = cmu.split_km(s)
    assert len(km.train) == 200 and len(km.genuine) == 200
    bp = cmu.split_bioprint(s, 10, "first")
    assert len(bp.train) == 10 and len(bp.genuine) == 350
    np.testing.assert_array_equal(bp.train, s.X[:10])
    last = cmu.split_bioprint(s, 10, "last")
    np.testing.assert_array_equal(last.train, s.X[40:50])


def test_eer_on_a_hand_made_case():
    # Perfectly separated: no threshold errs.
    assert cmu.eer([1.0, 2.0], [8.0, 9.0]) == 0.0
    # Identical distributions: every threshold errs on one side or the other; 50%.
    assert cmu.eer([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.5)
    # Genuine 0..9 against impostors 8..17. At threshold 8 exactly one genuine (9)
    # is rejected and exactly one impostor (8) is accepted -> FRR = FAR = 0.1.
    assert cmu.eer(np.arange(10.0), np.arange(8.0, 18.0)) == pytest.approx(0.1)


def test_frr_at_far_and_rates_at_threshold():
    g = np.arange(10.0)  # 0..9
    i = np.arange(100.0, 110.0)
    assert cmu.frr_at_far(g, i, 0.01) == 0.0  # separable: no genuine rejected at FAR 0
    # One impostor buried at 4.5: to keep FAR at 0 we must reject genuine 5..9.
    i2 = np.concatenate([[4.5], np.arange(100.0, 109.0)])
    assert cmu.frr_at_far(g, i2, 0.0) == pytest.approx(0.5)
    assert cmu.frr_at_far(g, i2, 0.1) == pytest.approx(0.0)  # 1 of 10 impostors accepted is allowed
    frr, far = cmu.rates_at(g, i2, 4.5)
    assert frr == pytest.approx(0.5)  # genuine 5..9 scored above the threshold
    assert far == pytest.approx(0.1)  # the buried impostor is accepted


def test_gate_a_bands():
    assert cmu.gate_a(0.05).startswith("PASS")
    assert cmu.gate_a(0.15).startswith("MARGINAL")
    assert cmu.gate_a(0.25).startswith("FAIL")


def test_end_to_end_on_synthetic_subjects(synthetic_csv):
    ds = cmu.load(synthetic_csv, synthetic_only=True)
    out = cmu.evaluate(ds, cmu.split_km)
    agg = out['aggregate']
    assert 0.0 <= agg["eer"]["mean"] <= 1.0


def test_real_all_session_loader_is_disabled():
    with pytest.raises(RuntimeError, match='test is sealed'):
        cmu.load(cmu.DATA)
