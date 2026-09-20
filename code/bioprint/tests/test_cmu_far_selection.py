"""Synthetic selector objective/tie checks, without dataset measurements."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from cmu_far_selection import METHODS, choose


def test_operational_frr_takes_priority_over_eer():
    values = {method: {'frr_at_selection_far_1pct': .8, 'eer': .2} for method in METHODS}
    values[METHODS[0]] = {'frr_at_selection_far_1pct': .5, 'eer': .01}
    values[METHODS[-1]] = {'frr_at_selection_far_1pct': .4, 'eer': .3}
    assert choose(values) == METHODS[-1]


def test_eer_then_fixed_grid_resolves_operational_ties():
    values = {method: {'frr_at_selection_far_1pct': .4, 'eer': .2} for method in METHODS}
    values[METHODS[2]]['eer'] = .1
    values[METHODS[3]]['eer'] = .1
    # Dictionary insertion order is irrelevant to the predeclared grid order.
    assert choose(dict(reversed(list(values.items())))) == METHODS[2]
