"""Export a completed frozen HMOG DEV result and replay it through the real API."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/benchmarks/hmog'
RELEASE = BASE / 'release_v1'
VALIDATION = BASE / 'validation_v1'
BRANCHES = ('joint', 'key', 'imu')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def export(calibration_dir, allow_incomplete_probes=False):
    if not calibration_dir.resolve().is_relative_to(ROOT):
        raise ValueError('Calibration artifacts must remain inside repository')
    report_bytes = (VALIDATION / 'validation_complete.json').read_bytes()
    report = json.loads(report_bytes)
    partial = (allow_incomplete_probes and report['status'] == 'infeasible'
               and bool(report['missing'])
               and all(row.get('missing_role') == 'dev_probe' for row in report['missing']))
    if ((report['status'] != 'complete' or report['missing']) and not partial
            or report['test_measurements_read'] is not False
            or report['validation_only'] is not True):
        raise ValueError('Complete four-account frozen DEV validation required')
    sources = {'encoder.pt': BASE / 'selection_v1/selected_checkpoint.pt',
               'profiles.npz': VALIDATION / 'profiles.npz',
               'thresholds.json': calibration_dir / 'thresholds.json',
               'dev_features.npz': VALIDATION / 'dev_features.npz'}
    expected = {'encoder.pt': report['plan']['selected_checkpoint_sha256'],
                'profiles.npz': report['profiles_sha256'],
                'thresholds.json': report['plan']['thresholds_sha256'],
                'dev_features.npz': report['dev_features_sha256']}
    blobs = {name: path.read_bytes() for name, path in sources.items()}
    if any(digest(raw) != expected[name] for name, raw in blobs.items()):
        raise ValueError('Frozen export artifact changed')
    # Do not expose support rows or silently drop an account from the release.
    import io
    accounts = sorted(('556357', '219303', '777078', '737973'))
    absent = {row['subject'] for row in report['missing']} if partial else set()
    if not absent.issubset(accounts) or len(absent) != len(report['missing']) or absent == set(accounts):
        raise ValueError('Invalid explicit missing-probe coverage')
    with np.load(io.BytesIO(blobs['dev_features.npz']), allow_pickle=False) as data:
        subjects, sessions, roles = (data[name].copy() for name in ('subject', 'session', 'role'))
        if (subjects.ndim != 1 or subjects.dtype.kind not in 'US'
                or sessions.shape != subjects.shape or sessions.dtype.kind not in 'iu'
                or roles.shape != subjects.shape or roles.dtype.kind not in 'US'
                or set(subjects.astype(str)) != set(accounts) - absent
                or not np.all(roles.astype(str) == 'dev_probe')
                or not np.all((sessions >= 9) & (sessions <= 16))):
            raise ValueError('Only four-account DEV probe rows may be released')
        key, imu = data['key'], data['imu']
        if (key.shape != (len(subjects), 50, 10) or imu.shape != (len(subjects), 100, 24)
                or key.dtype != np.float32 or imu.dtype != np.float32
                or not np.isfinite(key).all() or not np.isfinite(imu).all()):
            raise ValueError('Invalid frozen DEV feature contract')
    with np.load(io.BytesIO(blobs['profiles.npz']), allow_pickle=False) as data:
        if data['accounts'].astype(str).tolist() != accounts:
            raise ValueError('Profile order mismatch')
        for branch in BRANCHES:
            if data[branch].shape != (4, 64) or not np.isfinite(data[branch]).all():
                raise ValueError('Every account requires a finite profile')
    blobs['dev_report.json'] = report_bytes
    calibration_provenance = report['plan'].get('calibration_provenance', {})
    manifest = {'version': 1, 'split': 'dev',
                'validation_status': report['status'], 'missing': report['missing'],
                'coverage': report.get('coverage', {}),
                'all_accounts_have_probe_windows': not partial,
                'replay_scope': 'Observed DEV probes only; missing candidate windows remain abstentions',
                'files': {name: digest(raw) for name, raw in blobs.items()},
                'sources': {name: digest((ROOT / 'scripts' / name).read_bytes()) for name in
                            ('hmog_training_core.py', 'hmog_verification.py')},
                'export_source_sha256': digest(Path(__file__).read_bytes()),
                'calibration_scope': report['plan'].get('calibration_scope') or calibration_provenance.get('calibration_scope'),
                'calibration_limitation': calibration_provenance.get('limitation'),
                'calibration_protocol_sha256': report['plan'].get('calibration_protocol_sha256'),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'limitations': 'Small cohort; filtered real pairs; recording-time fusion; shared-checkpoint ablations'}
    RELEASE.mkdir(parents=True, exist_ok=False)
    for name, raw in blobs.items():
        (RELEASE / name).write_bytes(raw)
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    (RELEASE / 'manifest.json').write_bytes(raw)
    (RELEASE / 'export_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'manifest_sha256': digest(raw), 'release': str(RELEASE.relative_to(ROOT))}))


async def replay():
    import httpx
    import research_hmog_api
    from research_api import app
    from hmog_verification import pooled_eer, verification_rates

    manifest_bytes = (RELEASE / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != research_hmog_api.MANIFEST_SHA256:
        raise ValueError('API source must explicitly pin the exported release before replay')
    report_bytes = (VALIDATION / 'validation_complete.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if digest(report_bytes) != manifest['files']['dev_report.json']:
        raise ValueError('Offline validation report differs from frozen API release')
    report = json.loads(report_bytes)
    replay_path = VALIDATION / 'dev_replay.npz'
    if digest(replay_path.read_bytes()) != report['replay_sha256']:
        raise ValueError('Offline score archive changed')
    out = BASE / 'api_replay_v1'
    out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / 'code/bioprint/research_api.py', ROOT / 'code/bioprint/research_hmog_api.py',
             Path(__file__), RELEASE / 'manifest.json', replay_path,
             VALIDATION / 'validation_complete.json']
    before = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'sources_and_inputs': before,
            'validation_report_sha256': digest(report_bytes), 'batch_size': 8,
            'rtol': 1e-5, 'atol': 1e-7, 'decisions': 'exact at every frozen threshold',
            'test_measurements_read': False, 'listener_started': False}
    (out / 'replay_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'replay_source.py').write_bytes(Path(__file__).read_bytes())
    with np.load(replay_path, allow_pickle=False) as data:
        offline = {name: data[name].copy() for name in data.files}
    probe = offline['role'].astype(str) == 'dev_probe'
    key, imu, labels = offline['key'][probe], offline['imu'][probe], offline['subject'][probe].astype(str)
    accounts = offline['accounts'].astype(str).tolist()
    score_chunks = {branch: [] for branch in BRANCHES}
    max_error = {branch: 0. for branch in BRANCHES}
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 45123))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/hmog/test/samples')).status_code == 403
        for start in range(0, len(key), 8):
            response = await client.get('/api/hmog/dev/samples', params={'offset': start, 'limit': 8})
            response.raise_for_status()
            page = response.json()
            stop = min(start + 8, len(key))
            assert page['total'] == len(key) and page['accounts'] == accounts
            assert page['validation_status'] == report['status'] and page['missing'] == report['missing']
            assert page['coverage'] == report['coverage']
            assert page['subjects'] == labels[start:stop].tolist()
            np.testing.assert_array_equal(np.asarray(page['key'], np.float32), key[start:stop])
            np.testing.assert_array_equal(np.asarray(page['imu'], np.float32), imu[start:stop])
            response = await client.post('/api/hmog/score', json={'key': page['key'], 'imu': page['imu']})
            response.raise_for_status()
            result = response.json()
            assert result['accounts'] == accounts
            assert result['validation_status'] == report['status'] and result['missing'] == report['missing']
            for branch in BRANCHES:
                actual = np.asarray(result['branches'][branch]['distances'])
                expected = offline[branch + '_distances'][start:stop]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
                max_error[branch] = max(max_error[branch], float(np.max(np.abs(actual - expected))))
                assert result['branches'][branch]['thresholds'] == report['plan']['frozen_thresholds'][branch]
                for target in report['plan']['frozen_thresholds'][branch]:
                    np.testing.assert_array_equal(result['branches'][branch]['accepted'][target],
                                                  offline[branch + '_accepted_' + target][start:stop])
                score_chunks[branch].append(actual)
    metrics = {}
    genuine = offline['genuine_mask']
    for branch in BRANCHES:
        distances = np.concatenate(score_chunks[branch])
        g, i = distances[genuine], distances[~genuine]
        eer = pooled_eer(g, i)
        assert np.isclose(eer, report['metrics'][branch]['pooled_eer_diagnostic'], rtol=0, atol=1e-12)
        metrics[branch] = {'eer': eer, 'operating_points': {}}
        for target, threshold in report['plan']['frozen_thresholds'][branch].items():
            expected = report['metrics'][branch]['operating_points'][target]
            rates = verification_rates(g, i, threshold, expected_genuine=expected['expected_genuine'],
                                       expected_impostor=expected['expected_impostor'])
            assert rates == expected
            metrics[branch]['operating_points'][target] = rates
    assert before == {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}
    result = {'plan': plan, 'probe_windows': len(key), 'accounts': accounts,
              'validation_status': report['status'], 'missing': report['missing'],
              'coverage': report['coverage'],
              'branches': list(BRANCHES), 'scalar_scores': len(key) * len(accounts) * len(BRANCHES),
              'maximum_absolute_error': max_error, 'acceptance_flips': 0, 'metrics': metrics,
              'test_route_status': 403, 'listener_started': False, 'production_database_used': False}
    (out / 'replay_complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('plan', 'metrics')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('export', 'replay'))
    parser.add_argument('--calibration-dir', type=Path, default=BASE / 'calibration_fallback_v1')
    parser.add_argument('--allow-incomplete-probes', action='store_true',
                        help='Replay observed probes with explicit incomplete coverage; still require all galleries')
    args = parser.parse_args()
    export(args.calibration_dir, args.allow_incomplete_probes) if args.action == 'export' else asyncio.run(replay())
