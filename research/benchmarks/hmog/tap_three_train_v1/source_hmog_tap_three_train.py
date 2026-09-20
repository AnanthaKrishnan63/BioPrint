"""Frozen fixed-source-three-feature TRAIN diagnostic, no new feature selection.

Reuses the full11 diagnostic's exact first-session enrollment profiles. No raw,
DEV, test, population refitting, or independently optimized subset selection.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from hmog_tap_train import ROOT, PROTOCOL, digest, load, split, report_scores
from hmog_tap_three import FEATURE_INDICES, FEATURE_NAMES, score_windows
from hmog_tap_reference import validate_profile

OUT = ROOT / 'research/benchmarks/hmog/tap_three_train_v1'
ORIGINAL = ROOT / 'research/benchmarks/hmog/tap_train_v1'
ARCHIVE = ROOT / 'datasets/hmog/fit_tap_features_v1.npz'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    dependencies = [ROOT / 'scripts' / name for name in (
        'hmog_tap_three_train.py', 'hmog_tap_three.py', 'hmog_tap_train.py',
        'hmog_tap_benchmark.py', 'hmog_tap_reference.py', 'hmog_fit_tap_extract.py',
        'hmog_verification.py')]
    inputs = [PROTOCOL, ORIGINAL / 'profiles.json', ORIGINAL / 'complete.json', ARCHIVE]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in dependencies + inputs}
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Source-informed fixed3 TRAIN diagnostic; no DEV/raw/test reads',
            'feature_indices': list(FEATURE_INDICES), 'feature_names': list(FEATURE_NAMES),
            'source': 'https://arxiv.org/html/1501.01199#S4',
            'selection': 'Fixed published subset, not data-derived selection or mRMR reproduction',
            'prior_exposure': 'Full11 TRAIN diagnostic already run; no subset or numerical tuning from its scores',
            'profiles': 'Exact previously frozen individual-tap profiles from first eligible enrollment session',
            'scoring': 'Mean11D/window, sum standardized absolute differences on active0,1,10 only',
            'zero_spread': 'Ignore selected inactive dimensions; no verdict if all3 inactive',
            'counts': {'gallery_windows': 37, 'calibration_windows': 83, 'diagnostic_windows': 87},
            'source_and_input_sha256': hashes, 'dev_or_test_accessed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for path in dependencies:
        (OUT / ('source_' + path.name)).write_bytes(path.read_bytes())
    # All input/dependency pins and executable snapshots are durable before parsing.
    original = json.loads((ORIGINAL / 'complete.json').read_text())
    if original.get('status') != 'complete_train_diagnostic':
        raise ValueError('Complete original TRAIN diagnostic required')
    profiles = json.loads((ORIGINAL / 'profiles.json').read_text())
    for profile in profiles.values(): validate_profile(profile)
    protocol, meta, taps, index = load()
    accounts, gallery, calibration, diagnostic = split(meta)
    if set(profiles) != set(accounts):
        raise ValueError('Original galleries must preserve all four accounts')
    counts = {'gallery_windows': int(gallery.sum()), 'calibration_windows': int(calibration.sum()),
              'diagnostic_windows': int(diagnostic.sum())}
    if counts != plan['counts']:
        raise ValueError('Original window split changed')
    scored = score_windows(taps, index, len(meta['subject']), profiles, accounts)
    report = report_scores(scored['distances'], scored['available'], meta['subject'], accounts,
                           calibration, diagnostic, protocol['targets'])
    # Avoid object arrays: active counts are explicitly aligned to account order.
    arrays = {k: v for k, v in scored.items() if k != 'active_feature_counts'}
    arrays['active_feature_counts'] = np.array([scored['active_feature_counts'][a] for a in accounts])
    with (OUT / 'scores.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, gallery=gallery, calibration=calibration,
                            diagnostic=diagnostic, accounts=np.asarray(accounts), **meta)
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in dependencies + inputs}:
        raise ValueError('Frozen sources or inputs changed during diagnostic')
    result = {'status': 'complete_train_diagnostic', 'plan': plan, **counts,
              'active_feature_counts': scored['active_feature_counts'], 'tap3': report,
              'scores_sha256': digest(OUT / 'scores.npz'),
              'profiles_sha256': digest(ORIGINAL / 'profiles.json'),
              'dev_or_test_accessed': False,
              'interpretation': 'Source-subset adaptation on TRAIN; neither exact mRMR nor SOTA reproduction'}
    (OUT / 'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'tap3': report, 'active_feature_counts': scored['active_feature_counts'],
                      'scores_sha256': result['scores_sha256'],
                      'report_sha256': digest(OUT / 'complete.json'),
                      'plan_sha256': digest(OUT / 'plan.json')}, indent=2))


if __name__ == '__main__':
    main()
