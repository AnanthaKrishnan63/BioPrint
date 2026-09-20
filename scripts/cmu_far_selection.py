"""TRAIN-only objective-aligned per-account SVM selection at1% impostor FAR."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import numpy as np
from threadpoolctl import threadpool_limits
from eval import strict_cmu as cmu
from cmu_account_selection import METHODS, digest

OUT = ROOT / 'research/benchmarks/cmu-far-selection'
CONTROL = ROOT / 'research/benchmarks/cmu-account-selection/frozen_models.json'


def choose(values):
    return min(METHODS, key=lambda method: (values[method]['frr_at_selection_far_1pct'], values[method]['eer']))


def preregister():
    control = json.loads(CONTROL.read_text())
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Selecting for EER is mismatched to the primary operational goal of low FRR at low FAR; select directly for FRR at1% selection FAR.',
        'classification': 'objective-aligned adapted baseline optimization, not SOTA reproduction',
        'prior_exposure': 'Legacy CMU and previous global/EER-account development results already seen. This is exploratory, not independent confirmation. No new dev measurement is used for this selection rule.',
        'candidates_ordered': METHODS,
        'selection': 'Per account minimize genuine FRR at empirical impostor FAR<=1% on sessions2–3; derive candidate selection threshold using ONLY those sessions. Ties use EER then fixed listed grid order.',
        'enrollment': 'Exactly first10 repetitions of session1; no refit after selection',
        'feature_schema': 'Original strict CMU28 timing dimensions, milliseconds, Enter excluded; unchanged fit_model and distances',
        'background': 'Same other-account arrays from sessions1–2; SVM consumes first10 rows per account as negatives and population scaling',
        'calibration': 'Session4 only derives final EER/FAR1%/FAR5% operational thresholds; selection thresholds are never deployed.',
        'controls': ['baseline', 'svm-10-0.1', 'account_selected'],
        'original_selected': 'svm-10-0.1', 'control_artifact_sha256': digest(CONTROL),
        'strict_source_sha256': digest(Path(cmu.__file__)), 'script_sha256': digest(__file__),
        'cpu_threads': 2, 'runtime_estimate_seconds': '30–90 based on previous nine-candidate26.4second run',
        'training_gate': 'No dev evaluation if selected model is redundant or mean selection FRR fails to improve over EER-account control. Training-set advantage is selection-optimistic and not evidence of generalization.',
        'new_dev_measurements_parsed': False, 'test_measurements_parsed': False}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'preregistered.json').open('x') as f:
        json.dump(plan, f, indent=2)
    print('Objective and gate frozen before measurement access')


def train():
    plan = json.loads((OUT / 'preregistered.json').read_text())
    assert digest(CONTROL) == plan['control_artifact_sha256']
    assert digest(Path(cmu.__file__)) == plan['strict_source_sha256'] and digest(__file__) == plan['script_sha256']
    if (OUT / 'frozen_models.json').exists():
        raise SystemExit('Preserve completed experiment')
    control = json.loads(CONTROL.read_text())
    started = time.monotonic()
    names, records = cmu.load_partition('train')
    assert len(names) == 28
    values, candidates = {sid: {} for sid in records}, {}
    with threadpool_limits(limits=2):
        for method in METHODS:
            models = {}
            for sid, record in records.items():
                enrollment = record['X'][(record['session'] == 1) & (record['rep'] <= 10)]
                assert len(enrollment) == 10
                background = [r['X'][r['session'] <= 2] for owner, r in records.items() if owner != sid]
                models[sid] = cmu.fit_model(enrollment, names, method, background)
            for sid, (g, i) in cmu.scores_for(models, records, [2, 3]).items():
                threshold = cmu.threshold_for(g, i, .01)
                rates = cmu.rates(g, i, threshold)
                values[sid][method] = {'eer': cmu.eer(g, i), 'selection_threshold_not_deployed': threshold,
                                      'frr_at_selection_far_1pct': rates['frr'], 'achieved_selection_far': rates['far']}
            candidates[method] = models
            print(json.dumps({'candidate': method, 'macro_selection_frr_at_1pct': float(np.mean([v[method]['frr_at_selection_far_1pct'] for v in values.values()]))}), flush=True)
        selected = {sid: choose(row) for sid, row in values.items()}
        models = {sid: candidates[method][sid] for sid, method in selected.items()}
        for sid, (g, i) in cmu.scores_for(models, records, [4]).items():
            models[sid]['calibration'] = {key: cmu.threshold_for(g, i, target) for key, target in
                                        [('eer', None), ('far_1pct', .01), ('far_5pct', .05)]}
        selections = {'far_selected': selected, 'account_selected': control['selected_per_account'],
                      'svm-10-0.1': {sid: 'svm-10-0.1' for sid in records}}
        summary = {label: {metric: float(np.mean([values[sid][methods[sid]][metric] for sid in records]))
                          for metric in ['eer', 'frr_at_selection_far_1pct', 'achieved_selection_far']}
                   for label, methods in selections.items()}
        baseline_scores = cmu.scores_for(control['models']['baseline'], records, [2, 3])
        baseline_rows = [(cmu.eer(g, i), cmu.rates(g, i, cmu.threshold_for(g, i, .01))) for g, i in baseline_scores.values()]
        summary['baseline'] = {'eer': float(np.mean([e for e, _ in baseline_rows])),
                              'frr_at_selection_far_1pct': float(np.mean([r['frr'] for _, r in baseline_rows])),
                              'achieved_selection_far': float(np.mean([r['far'] for _, r in baseline_rows]))}
    changed = sum(selected[sid] != control['selected_per_account'][sid] for sid in selected)
    gain = summary['account_selected']['frr_at_selection_far_1pct'] - summary['far_selected']['frr_at_selection_far_1pct']
    result = {'protocol': plan, 'selected': 'far_selected', 'models': {**control['models'], 'far_selected': models},
              'selected_per_account': selected, 'training_selection_per_account': values,
              'training_summary': summary, 'selected_method_counts': dict(Counter(selected.values())),
              'changed_accounts_vs_eer_selection': changed, 'selection_frr_improvement_vs_eer': gain,
              'training_gate_passed': bool(changed > 0 and gain > 1e-12),
              'elapsed_seconds': time.monotonic() - started,
              'new_dev_measurements_parsed': False, 'test_measurements_parsed': False}
    cmu.write_json(OUT / 'frozen_models.json', result)
    (OUT / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({k: v for k, v in result.items() if k not in ['models', 'training_selection_per_account', 'protocol', 'selected_per_account']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preregister', 'train'], required=True)
    args = parser.parse_args()
    preregister() if args.stage == 'preregister' else train()
