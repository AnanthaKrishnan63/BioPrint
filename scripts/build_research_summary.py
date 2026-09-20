"""Collect frozen dev reports without loading models or dataset observations."""
import argparse
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/benchmarks'


def paired_type2branch_rows(result, path):
    """Summarize paired DEV evidence with denominator and per-person checks."""
    expected_models = {'typenet', 'sapimouse', 'handcrafted_behavior', 'old_dual_neural',
                       'old_hybrid', 'type2branch', 'type2branch_pointer', 'type2branch_hybrid'}
    if (result.get('status') != 'paired_dev_evaluation_complete'
            or result.get('evaluation_split') != 'dev'
            or result.get('test_observations_decoded') is not False
            or result.get('prior_dev_exposure') is not True
            or result.get('model_updates') != 0 or set(result['results']) != expected_models):
        raise ValueError('Completed frozen paired DEV evidence required')
    subjects = result['subjects']
    if not isinstance(subjects, list) or len(subjects) < 2 or len(set(subjects)) != len(subjects):
        raise ValueError('Distinct paired DEV identities required')
    for name in ['claims', 'windows', 'below25_windows']:
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError('Nonnegative integer paired counts required')
    if result['claims'] == 0 or not 0 <= result['below25_windows'] <= result['windows']:
        raise ValueError('Invalid paired coverage counts')

    def checked(metrics):
        n, m = metrics['genuine_comparisons'], metrics['impostor_comparisons']
        points = metrics['operating_points']['far_1pct']
        fr, fa = points['false_rejections'], points['false_acceptances']
        if (any(type(v) is not int for v in [n, m, fr, fa]) or n <= 0
                or m != n * (len(subjects) - 1) or not 0 <= fr <= n or not 0 <= fa <= m):
            raise ValueError('Paired claim counts disagree')
        for value in [metrics['eer_diagnostic'], points['far'], points['frr']]:
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Finite paired rates required')
        if abs(points['far'] - fa / m) > 1e-12 or abs(points['frr'] - fr / n) > 1e-12:
            raise ValueError('Paired rates disagree with counts')
        return n, m, fr, fa

    rows = []
    for model, values in result['results'].items():
        pooled = values['pooled']; n, m, fr, fa = checked(pooled)
        by_person = values['by_probe_identity']
        if set(by_person) != set(subjects) or n + m != result['claims']:
            raise ValueError('Complete paired cohort required')
        parts = [checked(v) for v in by_person.values()]
        if tuple(sum(p[i] for p in parts) for i in range(4)) != (n, m, fr, fa):
            raise ValueError('Pooled and per-person paired counts disagree')
        rows.append({'dataset': 'beacon-type2branch-paired-dev', 'model': model,
            'eer': pooled['eer_diagnostic'], 'eer_aggregation': 'pooled_discrete',
            'far': fa / m, 'frr': fr / n, 'genuine_n': n, 'impostor_n': m,
            'false_acceptances': fa, 'false_rejections': fr,
            'threshold_target_training_far': .01, 'source': path,
            'validation_status': result['status'], 'prior_dev_exposure': True,
            'dev_identities': len(subjects), 'paired_windows': result['windows'],
            'windows_below_source_training_minimum25': result['below25_windows'],
            'api_boundary': 'Frozen fusion on features from offline encoders; not live browser or encoder replay'})
    return rows


def capture_summary_entry(result, path):
    """Return one conditional row or an explicit no-observations status entry."""
    def rate(value):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Invalid capture rate')
        return float(value)

    def integer(value, maximum=None):
        if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
            raise ValueError('Invalid capture count')
        return value

    if (result.get('status') != 'frozen_capture_dev_evaluation_complete'
            or result.get('sealed_test_payloads_decoded') is not False):
        raise ValueError('Capture summary requires completed DEV-only evidence')
    exposure = result['prior_exposure']
    if not isinstance(exposure, list) or not exposure or not all(isinstance(s, str) and s for s in exposure):
        raise ValueError('Capture prior-exposure disclosure required')
    metrics = result['metrics']; coverage = metrics['coverage']; whole = metrics['whole_cohort']
    total = integer(coverage['total']); eligible = integer(coverage['eligible'], total)
    if total < 2 or abs(rate(coverage['fraction']) - eligible / total) > 1e-12:
        raise ValueError('Capture coverage fraction mismatch')
    missing = coverage['ineligible']
    if (integer(coverage['ineligible_count']) != total - eligible or not isinstance(missing, list)
            or len(missing) != total - eligible):
        raise ValueError('Capture unavailable-account count mismatch')
    names = []
    for item in missing:
        if (not isinstance(item, dict) or not isinstance(item.get('identity'), str)
                or not isinstance(item.get('reason'), str) or not item['reason']):
            raise ValueError('Capture unavailable-account reasons required')
        names.append(item['identity'])
    if len(set(names)) != len(names):
        raise ValueError('Duplicate unavailable account')
    direct = integer(whole['direct_acceptances'], eligible)
    rejected = integer(whole['score_rejections'], eligible)
    additional = integer(whole['additional_verifications'], total)
    if (integer(whole['total']) != total or integer(whole['ineligible']) != total - eligible
            or direct + rejected != eligible or additional != total - direct
            or abs(rate(whole['additional_verification_fraction']) - additional / total) > 1e-12):
        raise ValueError('Capture whole-cohort counts disagree')
    common = {'source': path, 'genuine_probe_coverage': eligible / total,
              'observed_genuine_probes': eligible, 'expected_candidate_genuine_probes': total,
              'whole_cohort_additional_verification_fraction': additional / total,
              'prior_dev_exposure': json.dumps(exposure, ensure_ascii=False)}
    if eligible == 0:
        if metrics['conditional_metrics'] != {}:
            raise ValueError('No eligible captures must have no recognition metrics')
        return None, {**common, 'experiment': 'type2branch-one-capture-dev',
            'status': 'no_eligible_captures', 'source_evaluation_status': result['status'],
            'reason': 'Evaluation completed with no score-eligible captures; additional verification for all',
            'recognition_metrics_available': False}
    pooled = metrics['conditional_metrics']['pooled']
    genuine = integer(pooled['genuine_n']); impostor = integer(pooled['impostor_n'])
    if genuine != eligible or impostor != eligible * (total - 1):
        raise ValueError('Capture all-claim denominators disagree with coverage')
    false_accepts = integer(pooled['false_acceptances'], impostor)
    false_rejects = integer(pooled['false_rejections'], genuine)
    far, frr = rate(pooled['far']), rate(pooled['frr'])
    if (abs(far - false_accepts / impostor) > 1e-12
            or abs(frr - false_rejects / genuine) > 1e-12 or false_rejects != rejected):
        raise ValueError('Capture conditional rates disagree with counts')
    return {**common, 'dataset': 'type2branch-one-capture-dev', 'model': 'train_selected_length_model',
        'eer': rate(pooled['discrete_eer']), 'eer_aggregation': 'pooled_raw_score_discrete_conditional',
        'margin_eer_diagnostic_additional': rate(pooled['margin_discrete_eer']),
        'far': far, 'frr': frr, 'genuine_n': genuine, 'impostor_n': impostor,
        'threshold_target_training_far': .01, 'threshold_scope': 'Frozen per exact calibrated probe length',
        'validation_status': result['status'], 'selected_checkpoint': json.dumps(result['selected_checkpoint'], sort_keys=True)}, None


def collect():
    rows, sources, infeasible = [], {}, []

    def read(path):
        raw = (BASE / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def add(dataset, model, eer, kind, rates, path):
        values = [eer, rates['far'], rates['frr']]
        if not all(0 <= float(value) <= 1 for value in values):
            raise ValueError(f'Invalid metric in {path}')
        rows.append({'dataset': dataset, 'model': model, 'eer': eer,
                     'eer_aggregation': kind, 'far': rates['far'], 'frr': rates['frr'],
                     'threshold_target_training_far': .01, 'source': path})

    for dataset in ['cmu', 'cmu100', 'cmu-account-selection', 'cmu-far-selection']:
        path = f'{dataset}/dev_results.json'
        for name, result in read(path)['results'].items():
            s = result['summary']
            add(dataset, name, s['macro_eer_diagnostic'], 'macro_per_account',
                s['operating_points']['far_1pct'], path)
    for track in ['fixed', 'free']:
        path = f'results/keyrecs-v1/{track}-results.json'
        for name, result in read(path)['validation'].items():
            s = result['aggregate_macro_user']
            add(f'keyrecs-{track}', name, s['eer'], 'macro_per_account', s, path)
    path = 'results/typenet-keyrecs-v1/dev-results.json'
    s = read(path)['validation']['aggregate_macro_user']
    add('keyrecs-free-typenet', 'frozen_typenet', s['eer'], 'macro_per_account', s, path)
    for dataset in ['device', 'delbot', 'cognitive', 'touch_tsi', 'beacon_nonlinear']:
        path = f'{dataset}_results.json'
        for name, result in read(path)['methods'].items():
            add(dataset, name, result['dev_eer_descriptive'], 'pooled',
                result['operating_points']['0.01']['dev'], path)
    for dataset in ['pointer', 'pointer_sapimouse', 'pointer_sapimouse_optimized']:
        path = f'{dataset}/results.json'
        for name, points in read(path)['dev'].items():
            s = points['0.01']
            add(dataset, name, s['eer_descriptive'], 'pooled', s, path)
            if 'macro_eer_descriptive' in s:
                rows[-1]['macro_eer_diagnostic_additional'] = s['macro_eer_descriptive']
    for dataset in ['beacon', 'beacon-neural-v1', 'beacon-neural-v2']:
        path = f'{dataset}/dev_results.json'
        for name, result in read(path)['results'].items():
            add(dataset, name, result['eer_diagnostic'], 'pooled',
                result['operating_points']['far_1pct'], path)
    path = 'hmog/validation_v1/validation_complete.json'
    if (BASE / path).exists():
        result = read(path)
        if not result.get('validation_only') or result.get('test_measurements_read') is not False:
            raise ValueError('HMOG summary requires frozen DEV-only evidence')
        for branch, metrics in result['metrics'].items():
            rates = metrics['operating_points']['0.01']
            if metrics['pooled_eer_diagnostic'] is None or rates['far'] is None or rates['frr'] is None:
                continue
            add('hmog-paired-v1', branch, metrics['pooled_eer_diagnostic'],
                'pooled_observed_claims_incomplete_cohort', rates, path)
            rows[-1].update({'validation_status': result['status'],
                            'genuine_probe_coverage': rates['genuine_coverage'],
                            'observed_genuine_probes': rates['genuine_count'],
                            'expected_candidate_genuine_probes': rates['expected_genuine'],
                            'genuine_not_accepted_rate_including_abstentions': rates['genuine_not_accepted_rate'],
                            'missing_roles': json.dumps(result['missing'], sort_keys=True),
                            'calibration_scope': result['plan'].get('calibration_scope') or
                                result['plan'].get('calibration_provenance', {}).get('calibration_scope')})
    path = 'pointer_sapimouse_znorm_dev_v2/dev_complete.json'
    if (BASE / path).exists():
        result = read(path)
        if (result.get('status') != 'complete_evaluation_only' or result.get('test_accessed') is not False
                or result.get('candidate_promoted') is not False):
            raise ValueError('Normalization summary requires frozen comparison-only evidence')
        for method, points in result['dev'].items():
            rates = points['0.01']
            add('pointer-sapimouse-znorm-comparison', method, rates['eer_descriptive'], 'pooled', rates, path)
            rows[-1].update({'validation_status': result['status'],
                            'training_selected_method': result['selected'],
                            'is_training_selected': method == result['selected'],
                            'prior_dev_exposure': result['plan']['prior_dev_exposure']})
    path = 'hmog/tap_dev_v1/dev_complete.json'
    if (BASE / path).exists():
        result = read(path)
        if result['status'] != 'infeasible' or result.get('metrics') != {}:
            raise ValueError('Inspect changed touch enrollment outcome before summarizing')
        infeasible.append({'experiment': 'hmog-touch-paired-dev', 'source': path,
                           'status': result['status'], 'reason': result['reason'],
                           'missing_gallery_accounts': result['missing_gallery_accounts'],
                           'recognition_metrics_available': False})
    path = 'arithmetic_dev_failure_v1/report.json'
    if (BASE / path).exists():
        result = read(path)
        if (result['status'] != 'infeasible' or result.get('metrics') != {}
                or result.get('test_accessed') is not False or result.get('subset_scored') is not False):
            raise ValueError('Inspect changed arithmetic outcome before summarizing')
        infeasible.append({'experiment': 'arithmetic-three-level-dev', 'source': path,
                           'status': result['status'], 'invalid_blocks': result['invalid_blocks'],
                           'training_selected_method': result['selected'],
                           'recognition_metrics_available': False})
    path = 'type2branch_dev_features_v1/report.json'
    if (BASE / path).exists():
        result = read(path)
        if (result['status'] != 'infeasible_prescribed_cohort' or result.get('metrics') != {}
                or result.get('sealed_test_payloads_decoded') is not False or not result.get('failures')):
            raise ValueError('Inspect changed Type2Branch outcome before summarizing')
        infeasible.append({'experiment': 'type2branch-cross-session-dev', 'source': path,
                           'status': result['status'], 'failures': result['failures'],
                           'recognition_metrics_available': False})
    path = 'type2branch_capture_dev_evaluation_v1/report.json'
    if (BASE / path).exists():
        row, unavailable = capture_summary_entry(read(path), path)
        if row is not None:
            rows.append(row)
        else:
            infeasible.append(unavailable)
    path = 'beacon_type2branch_dev_evaluation_v1/report.json'
    if (BASE / path).exists():
        rows.extend(paired_type2branch_rows(read(path), path))
    return {'rows': rows, 'source_sha256': sources, 'infeasible_experiments': infeasible,
            'constraints': ['Different datasets/tasks/enrollment budgets are not interchangeable.',
                            'EER aggregation differs; compare within the named protocol.',
                            'BEACON nonlinear wrappers retain original LR; interpolated EER differs from the original nearest-crossing convention, not model quality.',
                            'Operational thresholds are training-calibrated, not dev EER thresholds.',
                            'Some baselines change feature availability; inspect source limitations.',
                            'HMOG rows are conditional on observed probes; two DEV accounts have no eligible probes, full-cohort validation is infeasible, and calibration uses encoder-seen identities.',
                            'HMOG key/IMU branches share joint training and checkpoint selection; they are ablations, not independent optimized baselines.',
                            'SapiMouse Z-norm is an unselected frozen comparison, with earlier DEV exposure; its lower EER does not override the TRAIN FRR selection rule.',
                            'Infeasible enrollment experiments have no recognition metric rows; unavailable metrics are not zero errors.',
                            'Type2Branch capture rates are conditional on eligible first captures; raw-score EER and length-threshold-margin EER differ. The earlier fixed-window failure remains a separate experiment.',
                            'Paired Type2Branch transfer retains all short BEACON windows and previously exposed DEV identities. Its lower hybrid FAR trades against worse FRR/EER; offline encoders and fusion replay do not establish live all-feature verification.',
                            'No test observations or scores are read by this summary.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-stem', default='summary',
                        help='New report basename; existing artifacts are never overwritten')
    args = parser.parse_args()
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        parser.error('--output-stem must be a basename')
    report = collect()
    json_path = BASE / f'{args.output_stem}.json'
    csv_path = BASE / f'{args.output_stem}.csv'
    if json_path.exists() or csv_path.exists():
        raise SystemExit('Summary already exists; inspect logs before rebuilding')
    with json_path.open('x') as stream:
        json.dump(report, stream, indent=2)
    fields = list(dict.fromkeys(key for row in report['rows'] for key in row))
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(report['rows'])
    print(f"Wrote {len(report['rows'])} rows from {len(report['source_sha256'])} frozen reports")


if __name__ == '__main__':
    main()
