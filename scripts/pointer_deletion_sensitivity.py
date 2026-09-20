"""Leave-one-participant-out sensitivity of frozen saved dev scores; no tuning.

Remove the participant both as claimant and as probe owner, retaining the original
training-calibrated thresholds. Ranges describe cohort sensitivity, not confidence
intervals or independent-sample uncertainty. No raw measurement files are read.
"""
import json
from pathlib import Path
import numpy as np
from pointer_benchmark import stats
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/pointer_sapimouse_optimized'
result = json.loads((OUT / 'results.json').read_text())
frozen = json.loads((OUT / 'frozen_training_selection.json').read_text())
methods = {}
for name in result['dev']:
    with np.load(OUT / f'{name.replace(":", "_")}_dev_scores.npz', allow_pickle=False) as source:
        scores, users, actual = source['scores'], source['users'], source['true_user']
    threshold = frozen['thresholds'][name]['0.01']
    removals = {}
    for subject in users:
        surviving_users = users[users != subject]
        surviving_actual = actual[actual != subject]
        remaining = scores[users != subject][:, actual != subject]
        truth = np.asarray([surviving_actual == u for u in surviving_users]).ravel().astype(int)
        removals[str(subject)] = stats(truth, remaining.ravel(), threshold)
    ranges = {metric: {'min': min(r[metric] for r in removals.values()), 'max': max(r[metric] for r in removals.values())} for metric in ['far', 'frr', 'eer_descriptive']}
    methods[name] = {'original': {k:v for k,v in result['dev'][name]['0.01'].items() if k!='per_claimed_user'}, 'leave_one_participant_out_ranges': ranges, 'removed_participant_results': removals}
blocks = result['displacements_per_decision'] // 128
cost = json.loads((OUT.parent / 'pointer_sapimouse/training_observation_cost.json').read_text())[str(blocks)]
report = {'procedure': 'Delete each participant as both claimant and probe source; keep frozen training thresholds', 'interpretation': 'Cohort-deletion sensitivity ranges; not confidence intervals and not proof of low population FAR', 'methods': methods, 'selected_observation_cost_measured_training_only': cost, 'raw_measurements_accessed': False, 'model_or_threshold_changed': False}
(OUT / 'participant_deletion_sensitivity.json').write_text(json.dumps(report, indent=2))
print(json.dumps({name:r['leave_one_participant_out_ranges'] for name,r in methods.items()}, indent=2))
