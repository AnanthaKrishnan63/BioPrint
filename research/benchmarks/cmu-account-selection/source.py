"""TRAIN-only per-account SVM parameter selection; no development entry point."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from threadpoolctl import threadpool_limits
from eval import strict_cmu as cmu

OUT = ROOT / 'research/benchmarks/cmu-account-selection'
ORIGINAL = ROOT / 'research/benchmarks/cmu/frozen_models.json'
METHODS = [f'svm-{c}-{gamma}' for c in [.1, 1, 10] for gamma in [.1, 1, 10]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(values):
    """Fixed grid order resolves exact ties; no development-dependent preference."""
    return min(METHODS, key=lambda method: values[method])


def preregister():
    original = json.loads(ORIGINAL.read_text())
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Account-specific timing distributions may prefer different SVM regularization and kernel widths.',
        'classification': 'adapted baseline optimization; not SOTA reproduction',
        'historical_exposure': 'Legacy CMU sessions historically explored; earlier global-model dev summaries already seen. This follow-up is not independent confirmation and no pristine holdout claim is made.',
        'decision_scope': 'Existing TRAIN-only candidate grid; no candidate/threshold/feature choices from any new dev observations.',
        'candidates_ordered': METHODS, 'selection': 'Each account minimizes EER on sessions2and3; exact ties resolved by listed candidate order.',
        'enrollment': 'Exactly first10 repetitions of session1; no refit after selection',
        'feature_schema': 'Unchanged strict CMU schema excluding Enter, milliseconds; same fit_model and distances',
        'background': 'Unchanged original SVM implementation: other-account session1-2 arrays supplied, first10 rows per other account used as negative fit and population scaling.',
        'calibration': 'Session4 only; freeze account-specific EER/FAR1%/FAR5% thresholds using original threshold_for',
        'controls': ['original frozen baseline', 'original frozen globally selected SVM'],
        'original_selected': original['selected'], 'original_artifact_sha256': digest(ORIGINAL),
        'source_sha256': digest(Path(cmu.__file__)), 'script_sha256': digest(__file__),
        'training_budget': {'cpu_threads': 2, 'original19candidate_elapsed_seconds': original['elapsed_seconds'],
                            'estimated_seconds': '60–180 indicative, not measured; comfortably under10minutes'},
        'validation_condition': 'Stop after training and report. Any later dev comparison must use identical sessions5-6 and enrollment, frozen thresholds, paired account uncertainty; no automatic promotion from training EER.',
        'test_measurements_parsed': False, 'new_dev_measurements_parsed': False}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'preregistered.json').open('x') as f:
        json.dump(plan, f, indent=2)
    print('Plan frozen before TRAIN measurements are read')


def train():
    plan = json.loads((OUT / 'preregistered.json').read_text())
    assert digest(ORIGINAL) == plan['original_artifact_sha256']
    assert digest(Path(cmu.__file__)) == plan['source_sha256'] and digest(__file__) == plan['script_sha256']
    if (OUT / 'frozen_models.json').exists():
        raise SystemExit('Preserve frozen training result')
    original = json.loads(ORIGINAL.read_text())
    started = time.monotonic()
    names, records = cmu.load_partition('train')
    scores, candidates = {sid: {} for sid in records}, {}
    with threadpool_limits(limits=2):
        for method in METHODS:
            fitted = {}
            for sid, record in records.items():
                enrollment = record['X'][(record['session'] == 1) & (record['rep'] <= 10)]
                assert len(enrollment) == 10
                background = [r['X'][r['session'] <= 2] for other, r in records.items() if other != sid]
                fitted[sid] = cmu.fit_model(enrollment, names, method, background)
            for sid, (g, i) in cmu.scores_for(fitted, records, [2, 3]).items():
                scores[sid][method] = cmu.eer(g, i)
            candidates[method] = fitted
            print(json.dumps({'candidate': method, 'selection_macro_eer': float(np.mean([s[method] for s in scores.values()]))}), flush=True)
        selected = {sid: choose(values) for sid, values in scores.items()}
        models = {sid: candidates[method][sid] for sid, method in selected.items()}
        for sid, (g, i) in cmu.scores_for(models, records, [4]).items():
            models[sid]['calibration'] = {name: cmu.threshold_for(g, i, target) for name, target in
                                        [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
    result = {'protocol': plan, 'selected': 'account_selected',
              'models': {**original['models'], 'account_selected': models},
              'selected_per_account': selected, 'training_selection_per_account': scores,
              'training_summary': {'account_selected_macro_eer': float(np.mean([scores[sid][method] for sid, method in selected.items()])),
                  'global_control_macro_eer': float(np.mean([scores[sid][original['selected']] for sid in scores])),
                  'selected_method_counts': dict(Counter(selected.values())),
                  'selection_is_optimistic': 'Per-account minima necessarily look better on reused selection data; improvement is not a generalization claim.'},
              'elapsed_seconds': time.monotonic() - started,
              'new_dev_measurements_parsed': False, 'test_measurements_parsed': False}
    cmu.write_json(OUT / 'frozen_models.json', result)
    (OUT / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({**result['training_summary'], 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preregister', 'train'], required=True)
    args = parser.parse_args()
    preregister() if args.stage == 'preregister' else train()
