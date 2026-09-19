"""Scaled-Manhattan detector: shape, threshold rule, separation, explanation.

Accuracy itself is measured on CMU data by eval/cmu.py (Agent E). What is pinned
here is the behaviour the rest of the system relies on: a threshold that comes
from enrolment alone and stays inside its documented band, one wild feature not
being able to carry a verdict, old stored models still loading, and scoring being
fast enough to sit in a login round trip.
"""

import json
import random
import time

import pytest
from contracts import Sample
from engine import features as F
from engine import scorer
from helpers import CODES, make_sample


def vector(**kw) -> list[float]:
    s = Sample.model_validate(make_sample(**kw))
    return F.keystroke_vector(F.password_keystrokes(s)).values


def names() -> list[str]:
    return F.keystroke_vector(F.password_keystrokes(Sample.model_validate(make_sample(seed=0)))).names


def enrol(n=10, **kw) -> scorer.Model:
    return scorer.fit([vector(seed=i, **kw) for i in range(n)], names())


# ---------------------------------------------------------------- model shape

def test_fit_produces_a_usable_model():
    m = enrol()
    assert len(m.center) == len(m.spread) == len(m.names) == 3 * len(CODES) - 2
    assert m.n == 10
    assert all(s > 0 for s in m.spread)


def test_to_dict_from_dict_round_trip():
    m = enrol()
    back = scorer.Model.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.to_dict() == m.to_dict()
    assert scorer.distance(back, vector(seed=99))[0] == scorer.distance(m, vector(seed=99))[0]


def test_from_dict_loads_an_old_model_without_cap_or_loo():
    old = {"names": ["H.KeyT#0"], "center": [100.0], "spread": [10.0], "threshold": 1.5, "n": 10}
    m = scorer.Model.from_dict(old)
    assert m.cap == scorer.DEVIATION_CAP and m.loo == []
    assert scorer.distance(m, [110.0])[0] == pytest.approx(1.0)


def test_from_dict_ignores_keys_it_does_not_know():
    d = enrol().to_dict() | {"future_field": 42}
    assert scorer.Model.from_dict(d).n == 10


def test_wrong_length_vector_is_refused_rather_than_scored():
    # Failing closed: a shape mismatch means the template and the sample are not
    # the same password, and a silent 0 would be an "allow".
    with pytest.raises(ValueError):
        scorer.distance(enrol(), [1.0, 2.0])
    with pytest.raises(ValueError):
        scorer.fit([[1.0, 2.0]], ["only.one"])


def test_single_enrollment_sample_does_not_crash():
    m = scorer.fit([vector(seed=0)], names())
    assert all(s > 0 for s in m.spread)
    assert scorer.THRESHOLD_FLOOR <= m.threshold <= scorer.THRESHOLD_CEILING


# ---------------------------------------------------------------- threshold rule

def test_threshold_comes_from_enrollment_and_stays_in_its_band():
    m = enrol()
    assert scorer.THRESHOLD_FLOOR <= m.threshold <= scorer.THRESHOLD_CEILING
    assert len(m.loo) == 10  # one leave-one-out distance per enrolment sample
    assert m.loo == sorted(m.loo)


def test_leave_one_out_distances_are_honest():
    # Scored against a model that did NOT include the sample: they must be larger
    # than the in-sample distances, or the threshold is optimistic by construction.
    vs = [vector(seed=i) for i in range(10)]
    m = scorer.fit(vs, names())
    in_sample = [scorer.distance(m, v)[0] for v in vs]
    assert sum(m.loo) / 10 > sum(in_sample) / 10


def test_a_tidy_enrollment_gets_a_tighter_threshold_than_a_sloppy_one():
    tidy = scorer.fit([vector(seed=i, jitter=1.0) for i in range(10)], names())
    sloppy = scorer.fit([vector(seed=i, jitter=60.0) for i in range(10)], names())
    assert tidy.threshold <= sloppy.threshold
    assert sloppy.threshold == scorer.THRESHOLD_CEILING  # clipped: never trust a mess


def test_a_degenerate_enrollment_cannot_block_everyone():
    # Ten byte-identical samples (a replay, or a script): spread collapses to the
    # floors and the leave-one-out threshold to 0. Without the floor every later
    # login, including the owner's, would be rejected.
    same = [vector(seed=7)] * 10
    m = scorer.fit(same, names())
    assert m.threshold == scorer.THRESHOLD_FLOOR
    assert not scorer.score(m, vector(seed=7)).flagged


# ---------------------------------------------------------------- separation

def test_genuine_sample_scores_inside_the_threshold():
    m = enrol()
    for seed in (99, 100, 101, 102):
        r = scorer.score(m, vector(seed=seed))
        assert not r.flagged, (seed, r.score, r.threshold)


def test_a_different_rhythm_is_flagged():
    m = enrol()
    r = scorer.score(m, vector(hold=220, gap=400, seed=5))
    assert r.flagged and r.score > r.threshold


def test_separation_holds_over_many_impostors():
    m = enrol()
    rng = random.Random(0)
    genuine = [scorer.distance(m, vector(seed=1000 + i))[0] for i in range(20)]
    impostor = [scorer.distance(m, vector(hold=rng.uniform(140, 260), gap=rng.uniform(220, 450),
                                          jitter=20, seed=i))[0] for i in range(20)]
    assert max(genuine) < min(impostor)


def test_one_wild_feature_cannot_block_on_its_own():
    # A single feature 100 spreads out (the phone rang mid-password) is capped, so
    # it contributes DEVIATION_CAP / n_features and nothing more.
    m = enrol()
    v = list(vector(seed=99))
    v[0] += 100 * m.spread[0]
    r = scorer.score(m, v)
    assert not r.flagged
    assert r.contributions[0].deviation == pytest.approx(m.cap / len(m.names))


def test_a_feature_that_was_constant_in_enrollment_cannot_dominate():
    # Feature 0 identical in all ten enrolments; the spread floors stop its tiny,
    # meaningless variation from swamping every real feature at login.
    vs = [vector(seed=i) for i in range(10)]
    for v in vs:
        v[0] = 100.0
    m = scorer.fit(vs, names())
    v = list(vector(seed=99))
    v[0] = 103.0  # 3 ms off a "perfectly consistent" feature
    assert not scorer.score(m, v).flagged


def test_score_is_comparable_across_password_lengths():
    # The dashboard prints score/threshold for everyone, so the unit must not grow
    # with the number of features: same rhythm, 4 keys vs 12, same neighbourhood.
    long_codes = CODES * 3
    short = scorer.fit([vector(seed=i) for i in range(10)], names())
    long_names = F.keystroke_vector(F.password_keystrokes(
        Sample.model_validate(make_sample(codes=long_codes, seed=0)))).names
    long = scorer.fit([vector(codes=long_codes, seed=i) for i in range(10)], long_names)
    # Averaged over 20 logins: one 10-feature sample is far too noisy to compare.
    a = sum(scorer.distance(short, vector(seed=99 + i))[0] for i in range(20)) / 20
    b = sum(scorer.distance(long, vector(codes=long_codes, seed=99 + i))[0] for i in range(20)) / 20
    assert abs(a - b) < 0.3, (a, b)
    assert abs(short.threshold - long.threshold) < 0.5


def test_distance_shares_sum_to_the_score():
    m = enrol()
    total, share = scorer.distance(m, vector(seed=99))
    assert share.sum() == pytest.approx(total)
    assert len(share) == len(m.names)


# ---------------------------------------------------------------- explanation

def test_contributions_report_the_enrolled_expectation():
    m = enrol()
    v = vector(seed=99)
    r = scorer.score(m, v)
    assert 1 <= len(r.contributions) <= scorer.MAX_CONTRIBUTIONS
    c = r.contributions[0]
    i = m.names.index(c.feature)
    assert c.value == v[i] and c.expected == m.center[i]
    assert [c.deviation for c in r.contributions] == sorted((c.deviation for c in r.contributions), reverse=True)


def test_reasons_are_plain_english_about_the_right_key():
    m = enrol()
    v = list(vector(seed=99))
    v[m.names.index("H.KeyT#0")] += 300  # held T far too long
    reasons = scorer.score(m, v).reasons
    assert any("held 'T'" in r and "longer" in r for r in reasons), reasons
    assert not any("#" in r or "H." in r for r in reasons), reasons


def test_reasons_name_a_pause_and_a_slow_digraph():
    m = enrol()
    v = list(vector(seed=99))
    v[m.names.index("UD.KeyE#2.Digit5#3")] += 400
    v[m.names.index("DD.KeyE#2.Digit5#3")] += 400
    reasons = scorer.score(m, v).reasons
    assert any("paused" in r and "'5'" in r for r in reasons), reasons
    assert any("pressed '5'" in r and "later" in r for r in reasons), reasons


def test_reasons_fall_back_to_a_generic_sentence_for_other_signals():
    # The same scorer runs on pointer vectors, whose names carry no key structure.
    m = scorer.fit([[0.9 + i * 0.01, 300.0 + i] for i in range(10)],
                   ["pointer.path_efficiency", "pointer.hover_ms"])
    r = scorer.score(m, [0.2, 900.0], name="pointer")
    assert r.name == "pointer"
    assert any("pointer.path_efficiency was 0.2" in x for x in r.reasons), r.reasons


def test_an_unremarkable_sample_still_explains_itself():
    m = enrol()
    r = scorer.score(m, vector(seed=99))
    assert r.reasons and all(isinstance(x, str) and x for x in r.reasons)
    assert len(r.reasons) <= scorer.MAX_REASONS


def test_a_diffuse_mismatch_says_so():
    # Every feature moved a little, none of them dramatically: there is no single
    # honest sentence to give, and pretending otherwise would be misleading.
    m = enrol()
    # Just over the threshold, just under "worth a sentence" for any one feature.
    dev = (m.threshold + scorer.REASON_MIN_DEVIATION) / 2
    assert m.threshold < dev < scorer.REASON_MIN_DEVIATION, "constants no longer leave a diffuse band"
    v = [c + dev * s for c, s in zip(m.center, m.spread)]
    r = scorer.score(m, v)
    assert r.flagged
    assert any("small timing differences" in x for x in r.reasons), r.reasons


# ---------------------------------------------------------------- latency

def test_scoring_is_well_under_five_milliseconds():
    m = enrol()
    v = vector(seed=99)
    scorer.score(m, v)  # warm up numpy
    t0 = time.perf_counter()
    for _ in range(50):
        scorer.score(m, v)
    per_call_ms = (time.perf_counter() - t0) * 1000 / 50
    assert per_call_ms < 5.0, per_call_ms


def test_fitting_ten_samples_is_fast():
    t0 = time.perf_counter()
    enrol()
    assert (time.perf_counter() - t0) * 1000 < 100
