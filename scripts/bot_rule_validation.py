"""Frozen production bot-rule specificity audit on strict CMU dev timings.

No training/tuning, no database, no listener, no invented pointer/browser probe.
Synthetic enrollment jitter checks are reported separately from genuine dev rates.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
from contracts import Sample
from engine import bot
from engine.features import keystroke_vector, password_keystrokes
from eval.strict_cmu import load_partition
from production_api_replay import sample as reconstruct

OUT = ROOT / 'research/benchmarks/bot_rules'
ALL_RULES = ['untrusted_keys', 'untrusted_pointer', 'no_probe', 'webdriver', 'automation_globals', 'headless_ua', 'software_gl', 'zero_outer_window', 'no_plugins', 'no_languages', 'chrome_object_missing', 'permission_mismatch', 'mobile_no_touch', 'single_core', 'paste', 'no_typing', 'too_few_keys', 'impossible_hold', 'impossible_dd', 'identical_dd', 'identical_hold', 'low_variance_dd', 'low_variance_hold', 'round_timings', 'replay', 'click_without_pointer', 'click_without_movement', 'pointer_teleport', 'probe_error']
OBSERVED_TIMING_RULES = ['impossible_hold', 'impossible_dd', 'identical_dd', 'identical_hold', 'low_variance_dd', 'low_variance_hold', 'round_timings', 'replay']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(names, row):
    record = Sample.model_validate(reconstruct(names, row))
    vector = keystroke_vector(password_keystrokes(record))
    return record, vector


def evaluate(record, vector, prior):
    signal = bot.check(record, vector, prior, password_length=len('.tie5Roanl'))
    rules = [part.feature.removeprefix('bot.') for part in signal.contributions]
    assert record.env == {} and record.pointer == []
    return {'flagged': bool(signal.flagged), 'score': float(signal.score), 'rules': rules}


def summarize(rows):
    counts = Counter(rule for row in rows for rule in row['rules'])
    flagged = sum(row['flagged'] for row in rows)
    return {'samples': len(rows), 'flagged': flagged, 'flag_rate': flagged / len(rows), 'per_rule_count': {rule: counts[rule] for rule in ALL_RULES}, 'score_counts': dict(Counter(str(row['score']) for row in rows))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise SystemExit('Existing rule validation found; inspect logs rather than rerunning')
    constants = {name: getattr(bot, name) for name in ['THRESHOLD', 'STRONG', 'HOLD_FLOOR_MS', 'DD_FLOOR_MS', 'DD_FLOOR_FRACTION', 'MIN_CV', 'MIN_TIMING_N', 'IDENTICAL_TOL_MS', 'ROUND_GRID_MS', 'REPLAY_MAE_MS', 'TELEPORT_PX']}
    plan = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'source': 'strict CMU partition loader', 'training_prior': 'first10repetitions of session1 per account, reconstructed through production event/feature functions', 'genuine_dev': 'every session5/6 row, checked against only its own account training prior', 'sealed_test': 'sessions7/8 skipped before timing parsing', 'browser_environment': {}, 'pointer': [], 'submission': 'Enter', 'trust_assumption': 'All reconstructed events trusted=True; source has no browser isTrusted field, so trust rules cannot be validated', 'synthetic_stress': {'source': 'first10training enrollment event streams per subject', 'timestamp_jitter_ms': [0, 2, 5, 10], 'distribution': 'independent uniform[-jitter,+jitter] per event, then time-sort', 'seed': 20260920, 'classification': 'synthetic input robustness checks, never real heldout bot FAR'}, 'rules_frozen': constants, 'code_sha256': {'script': digest(__file__), 'bot': digest(ROOT/'code/bioprint/engine/bot.py'), 'reconstruction': digest(ROOT/'scripts/production_api_replay.py'), 'features': digest(ROOT/'code/bioprint/engine/features.py')}, 'tuning': 'none; no changes to rules, thresholds, templates or model selection from dev'}
    plan_path = OUT / 'frozen_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        assert previous['code_sha256'] == plan['code_sha256'], 'Frozen validation code changed'
        plan = previous
    else:
        # This write must precede every dataset measurement read below.
        plan_path.write_text(json.dumps(plan, indent=2))
    started = time.time()
    names, train = load_partition('train')
    priors, enrollment = {}, {}
    for owner, source in train.items():
        indices = np.flatnonzero(source['session'] == 1)
        indices = indices[np.argsort(source['rep'][indices])][:10]
        pairs = [prepare(names, row) for row in source['X'][indices]]
        assert len(pairs) == 10
        priors[owner] = [vector.values for _, vector in pairs]
        enrollment[owner] = [record for record, _ in pairs]
    dev_names, dev = load_partition('dev')
    assert dev_names == names
    observations = []
    max_reconstruction_difference = 0.
    for owner, source in dev.items():
        for session, repetition, values in zip(source['session'], source['rep'], source['X']):
            record, vector = prepare(names, values)
            named = dict(zip(vector.names, vector.values))
            difference = max(abs(named[name] - value) for name, value in zip(names, values))
            max_reconstruction_difference = max(max_reconstruction_difference, difference)
            observation = evaluate(record, vector, priors[owner])
            observations.append({'subject': owner, 'session': int(session), 'repetition': int(repetition), **observation})
    genuine = summarize(observations)
    per_account = {owner: summarize([r for r in observations if r['subject'] == owner]) for owner in sorted(dev)}
    flagged_accounts = sum(r['flagged'] > 0 for r in per_account.values())
    # Do not tune from genuine results; this exact synthetic grid was frozen above.
    rng = np.random.default_rng(plan['synthetic_stress']['seed'])
    synthetic = {}
    for jitter in [0, 2, 5, 10]:
        stress_rows = []
        for owner in sorted(enrollment):
            for repetition, original in enumerate(enrollment[owner], 1):
                record = original.model_copy(deep=True)
                if jitter:
                    for event in record.keystrokes:
                        event.t += float(rng.uniform(-jitter, jitter))
                    record.keystrokes.sort(key=lambda e: e.t)
                vector = keystroke_vector(password_keystrokes(record))
                stress_rows.append({'subject': owner, 'source_training_repetition': repetition, **evaluate(record, vector, priors[owner])})
        synthetic[str(jitter)] = {'summary': summarize(stress_rows), 'observations': stress_rows}
    result = {'evaluation': 'Frozen production bot-rule false flags on genuine CMU dev timing reconstructions', 'plan_sha256': digest(plan_path), 'genuine_dev': genuine, 'account_clusters': {'accounts': len(per_account), 'accounts_with_any_flag': flagged_accounts, 'fraction_with_any_flag': flagged_accounts / len(per_account), 'macro_account_false_flag_rate': float(np.mean([r['flag_rate'] for r in per_account.values()])), 'per_account': per_account}, 'observations': observations, 'reconstruction_max_abs_feature_difference_ms': float(max_reconstruction_difference), 'synthetic_training_clone_stress': synthetic, 'elapsed_seconds': time.time() - started, 'limitations': ['CMU rules were historically developed using this legacy corpus; this is not independent or pristine holdout specificity.', 'Source lacks isTrusted: true is an explicit reconstruction assumption; zero trust flags are not evidence of trust-rule validity.', 'Empty environment contributes constant weak no_probe=0.4; no browser probe/device/pointer telemetry was invented.', 'Only timing and replay behavior is observed; environment/pointer/trust bot detection cannot be estimated from this source.', 'Synthetic training clones and jitter are not real bots or heldout attacks; flagged fractions are not real-world bot FAR.', 'No bot-positive public cohort exists here, so no bot FAR/EER is claimed.', 'Reconstructed timing features do not reproduce all browser sampling/event-delivery behavior.', 'No database, production user recordings, network listener or sealed test timing values accessed.']}
    (OUT/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'genuine_dev': genuine, 'accounts_with_any_flag': flagged_accounts, 'accounts': len(per_account), 'reconstruction_max_abs_feature_difference_ms': max_reconstruction_difference, 'synthetic_stress_flag_rates': {j: r['summary']['flag_rate'] for j,r in synthetic.items()}}, indent=2))

if __name__ == '__main__':
    main()
