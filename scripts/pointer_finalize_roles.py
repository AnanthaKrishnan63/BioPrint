"""Finalize metadata-only support-role clarification; no measurement or score reads.

The running original FCN job had cached cohort labels before support files were
explicitly marked training. Its suffix-based fitting used exactly those support
files. Preserve original reports and record the declaration timing for audit.
"""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/pointer_sapimouse'
manifest = json.loads((ROOT / 'data/benchmarks/sapimouse/split_manifest.json').read_text())
marker = OUT / 'support_role_clarification.json'
if not (OUT / 'results.json').exists():
    raise SystemExit('Training/validation has not finished')
if (OUT / 'role_finalization.json').exists():
    raise SystemExit('Metadata already finalized; no changes made')
assert marker.stat().st_mtime < (OUT / 'frozen_training_selection.json').stat().st_mtime
provenance = json.loads((OUT / 'run_provenance.json').read_text())
assert hashlib.sha256((OUT / 'executed_initial_script.py').read_bytes()).hexdigest() == provenance['files_sha256']['scripts/pointer_sapimouse_benchmark.py']
records = {'files_by_split': dict(collections.Counter(e['split'] for e in manifest['files'])), 'files_by_role': dict(collections.Counter(e['role'] for e in manifest['files']))}
evidence = {'support_roles_declared_utc': datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc).isoformat(), 'training_selection_frozen_utc': datetime.fromtimestamp((OUT / 'frozen_training_selection.json').stat().st_mtime, timezone.utc).isoformat(), 'record_roles': records, 'selection_thresholds_and_scores_changed': False, 'model_fitting_membership_changed': False}
for name in ['frozen_training_selection', 'results']:
    path = OUT / f'{name}.json'
    content = path.read_text()
    (OUT / f'{name}.initial_cohort_labels.json').write_text(content)
    result = json.loads(content)
    result['protocol'] = manifest['protocol']
    result['record_roles'] = records
    result['support_role_evidence'] = evidence
    if name == 'results':
        if 'dev_enroll_windows' in result['counts']:
            result['counts']['training_support_enrollment_windows'] = result['counts'].pop('dev_enroll_windows')
        result['limitations'].append('The24unseen-encoder dev-cohort profiles fit only3min TRAINING-support records;1min DEV-probe records never fit or scale models.')
    path.write_text(json.dumps(result, indent=2))
(OUT / 'role_finalization.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
