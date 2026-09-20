"""Freeze a disclosed short-capture DEV follow-up without reading observations."""
import datetime
import json
from pathlib import Path
from type2branch_continuity import ROOT, DATA, sha

OUT = ROOT/'research/benchmarks/type2branch_capture_protocol_v1'


def main():
    roles_path = ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    split_path = DATA/'split-manifest.json'
    roles = json.loads(roles_path.read_text())['roles']
    split = json.loads(split_path.read_text())
    allowed = set(split['active_subjects'])
    groups = [set(roles[name]) for name in ['fit', 'selection', 'calibration']]
    if (len(allowed) != 79 or allowed & set(split['sealed_test_subjects'])
            or set.union(*groups) != allowed or sum(map(len, groups)) != len(allowed)):
        raise ValueError('Expected disjoint complete active identity roles')
    failed = ROOT/'research/benchmarks/type2branch_dev_features_v1/report.json'
    failure = json.loads(failed.read_text())
    if failure['status'] != 'infeasible_prescribed_cohort' or failure['metrics'] != {}:
        raise ValueError('Original failed DEV record changed')
    feature_plan = ROOT/'research/benchmarks/type2branch_length_calibration_features_v1/plan.json'
    if json.loads(feature_plan.read_text())['lengths'] != [25, 50, 75, 100]:
        raise ValueError('Capture anchors must match frozen calibration lengths')
    files = [Path(__file__), roles_path, split_path, failed, feature_plan]
    files += [ROOT/'scripts'/name for name in ['type2branch_capture_policy.py',
        'type2branch_variable_adapter.py', 'type2branch_keyrecs_adapter.py',
        'type2branch_context_model.py', 'type2branch_random.py',
        'type2branch_residual_features.py', 'type2branch_synthesis_cleanup.py',
        'type2branch_csv_bridge.py', 'type2branch_scoring.py']]
    protocol = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Disclosed cross-session short-capture DEV follow-up, not replacement of failed fixed-capture v1',
        'allowed_identities': sorted(allowed),
        'cohorts': {name: sorted(roles[name]) for name in ['fit', 'selection', 'calibration']},
        'gallery': {'session': '1', 'windows': [15, 16, 17, 18, 19],
                    'events_per_window': 100, 'required_all_identities': True},
        'probe': {'session': '2', 'captures_per_identity': 1, 'maximum_events': 100,
                  'anchors': [25, 50, 75, 100],
                  'allocation': 'Largest anchor <= available events; only that initial prefix is parsed and synthesized',
                  'below_minimum': 'Additional verification; retained in whole-cohort coverage and action counts'},
        'synthesis': 'Frozen fit-only Average population; separate fresh first-thread gallery and probe RNG streams, sorted identity/window order; real events before padding',
        'model': 'Exactly the completed paired TRAIN comparison selected checkpoint; no DEV selection',
        'threshold': 'Exact-length threshold from completed reserved-TRAIN calibration; inclusive score>=threshold; no DEV retuning',
        'claims': 'Every eligible probe against all 79 galleries; one genuine and 78 impostor claims per eligible identity',
        'failure_policy': 'Any missing gallery or malformed required prefix makes the prescribed experiment infeasible; no replacement, identity drop, or subset performance report',
        'reporting': {
            'coverage': 'Eligible/79 overall and by TRAIN role and length; all ineligible identities and reasons retained',
            'conditional_metrics': 'FAR/FRR/discrete EER only among score-eligible captures, explicitly labeled conditional',
            'whole_cohort_actions': 'Report genuine direct-accept fraction and additional-verification fraction with denominator79; do not label unscored captures model rejects',
            'cohorts': 'Full claim pool grouped by probe TRAIN role, plus within-role comparisons where defined',
            'lengths': 'Report counts and eligible conditional metrics by exact length; never choose a length based on DEV performance',
            'all_ineligible': 'Coverage0, empty model metrics and additional verification for all; no artificial perfect FAR'},
        'prior_exposure': [
            'Original fixed1000event DEV preparation failed because p004 S2 has38publishedrows',
            'Earlier DEV preparation parsed prescribed non-test prefixes and prepared79galleries; no completed encoder DEV scoring',
            'Other KeyRecs experiments already used this DEV split; this is not pristine or final-test evaluation',
            'TRAIN short-length sensitivity motivated paired feature-prefix augmentation and raw-prefix calibration'],
        'limitations': [
            'Different capture budget and trial count from failed original protocol; no direct performance comparison',
            'Prefix anchoring discards at most24 events within the first100; later events unused',
            'One chronological probe per identity gives limited precision and dependent all-claim comparisons',
            'Same-session TRAIN calibration does not guarantee cross-session FAR',
            'Source-order, Average synthesis and residual preprocessing are documented adaptations',
            'No all-feature fusion claim; this experiment measures the keystroke encoder'],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}}
    OUT.mkdir(exist_ok=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': 'protocol_frozen', 'sha256': sha(OUT/'protocol.json'),
                      'identities': 79, 'maximum_probes': 79, 'payloads_read': False}))


if __name__ == '__main__':
    main()
