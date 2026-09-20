"""Check all Balabit dev decisions before runtime-only ExtraTrees thread change."""
import copy
import json
from pathlib import Path
import joblib
import numpy as np
from pointer_benchmark import load_entries, DATA, OUT

manifest = json.loads((DATA / 'split_manifest.json').read_text())
x, labels, groups = load_entries([e for e in manifest['entries'] if e['split'] == 'dev'])
frozen = json.loads((OUT / 'frozen_training_selection.json').read_text())
method = frozen['selected']
with np.load(OUT / f'{method}_dev_scores.npz', allow_pickle=False) as source:
    users = source['users'].tolist()
    expected = source['scores']
results = {}
for index, user in enumerate(users):
    original = joblib.load(OUT / f'{method}_{user}.joblib')
    single = copy.deepcopy(original).set_params(n_jobs=1)
    original_scores = original.predict_proba(x)[:, 1]
    single_scores = single.predict_proba(x)[:, 1]
    np.testing.assert_allclose(original_scores, expected[index], rtol=0, atol=1e-15)
    np.testing.assert_allclose(single_scores, original_scores, rtol=0, atol=1e-15)
    for threshold in frozen['thresholds'][method].values():
        np.testing.assert_array_equal(single_scores >= threshold, original_scores >= threshold)
    results[user] = {'original_n_jobs': original.n_jobs, 'candidate_n_jobs': single.n_jobs, 'dev_windows': len(x), 'maximum_absolute_probability_difference': float(np.abs(single_scores-original_scores).max()), 'all_frozen_operating_point_decisions_identical': True}
report = {'method': method, 'claims': len(users), 'dev_windows': len(x), 'claim_window_comparisons': len(users)*len(x), 'threshold_targets': list(frozen['thresholds'][method]), 'per_claim': results, 'model_training_or_thresholds_changed': False, 'runtime_only_change_approved_by_exact_decision_parity': True}
(OUT.parent / 'pointer_runtime_thread_parity.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
