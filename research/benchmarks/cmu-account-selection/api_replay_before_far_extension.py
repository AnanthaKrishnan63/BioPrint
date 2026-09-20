"""Replay frozen CMU account-selection dev through real ASGI routes, no listener.

Scores are negative distances, not probabilities. No fitting or test reads.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import httpx
import numpy as np
API_SOURCE_PATH = ROOT / 'code/bioprint/research_api.py'
API_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(API_SOURCE_PATH.read_bytes()).hexdigest()
from research_api import app
from eval.strict_cmu import eer, rates

DIRECTORY = ROOT / 'research/benchmarks/cmu-account-selection'
DATASET = 'cmu-account-selection'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_metrics(labels, distances, subjects, frozen_models, saved):
    """Evaluate API-returned scores using the identical strict-CMU convention."""
    per_user = {}
    for column, subject in enumerate(subjects):
        g, i = distances[labels == column, column], distances[labels != column, column]
        actual = {'eer_diagnostic': eer(g, i),
                  'operating_points': {point: rates(g, i, threshold)
                                       for point, threshold in frozen_models[subject]['calibration'].items()}}
        expected = saved['per_user'][subject]
        if actual['eer_diagnostic'] != expected['eer_diagnostic']:
            raise AssertionError(f'{subject}: strict-CMU EER differs')
        for point, values in actual['operating_points'].items():
            if values != expected['operating_points'][point]:
                raise AssertionError(f'{subject}/{point}: rates differ')
        per_user[subject] = actual
    summary = {'macro_eer_diagnostic': float(np.mean([per_user[s]['eer_diagnostic'] for s in subjects])),
               'operating_points': {point: {rate: float(np.mean([
                    per_user[s]['operating_points'][point][rate] for s in subjects]))
                    for rate in ['far', 'frr']} for point in ['eer', 'far_1pct', 'far_5pct']}}
    expected = saved['summary']
    assert summary['macro_eer_diagnostic'] == expected['macro_eer_diagnostic']
    assert summary['operating_points'] == expected['operating_points']
    return summary


async def main():
    destination = DIRECTORY / 'api_replay.json'
    if destination.exists():
        raise ValueError('Existing replay report preserved')
    paths = [DIRECTORY / name for name in ['frozen_models.json', 'dev_results.json', 'dev_scores.npz']]
    original_hashes = {p.name: digest(p) for p in paths}
    artifact = json.loads(paths[0].read_text())
    offline_report = json.loads(paths[1].read_text())
    assert original_hashes['frozen_models.json'] == offline_report['artifact_sha256']
    assert original_hashes['dev_scores.npz'] == offline_report['dev_scores_sha256']
    with np.load(paths[2], allow_pickle=False) as archive:
        subjects = archive['subjects'].tolist()
        feature_names = archive['feature_names'].tolist()
        labels = archive['y'].copy()
        aliases = {'baseline': 'baseline', 'train_selected': artifact['selected'],
                   'global_reference': artifact['protocol']['original_selected']}
        offline_scores = {alias: -archive[method].copy() for alias, method in aliases.items()}
    assert len(labels) == 5100 and len(subjects) == 51
    started = time.monotonic()
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
    results = {}
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1', timeout=120) as client:
        for suffix in ['samples', 'raw']:
            response = await client.get(f'/api/datasets/{DATASET}/test/{suffix}')
            assert response.status_code == 403
        x, served_labels = [], []
        while len(x) < len(labels):
            response = await client.get(f'/api/datasets/{DATASET}/dev/samples',
                                        params={'offset': len(x), 'limit': 256})
            response.raise_for_status()
            page = response.json()
            assert page['subjects'] == subjects and page['feature_names'] == feature_names
            assert page['total'] == len(labels) and page['offset'] == len(x)
            assert page['x'], 'Pagination made no progress'
            x.extend(page['x'])
            served_labels.extend(page['y'])
        np.testing.assert_array_equal(served_labels, labels)
        assert len(x) == len(labels)
        for alias, method in aliases.items():
            assert sorted(artifact['models'][method]) == subjects
            thresholds = np.array([-artifact['models'][method][s]['calibration']['far_1pct'] for s in subjects])
            all_scores, all_decisions = [], []
            for start in range(0, len(x), 256):
                response = await client.post(f'/api/models/{DATASET}/score',
                    json={'model': alias, 'features': x[start:start+256]})
                response.raise_for_status()
                body = response.json()
                assert body['subjects'] == subjects
                assert body['score_direction'] == 'larger_is_genuine'
                np.testing.assert_array_equal(body['thresholds'], thresholds)
                batch_scores = np.array(body['scores'])
                batch_decisions = np.array(body['accepted'], dtype=bool)
                expected = offline_scores[alias][start:start+len(batch_scores)]
                np.testing.assert_allclose(batch_scores, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(batch_decisions, batch_scores >= thresholds)
                np.testing.assert_array_equal(batch_decisions, expected >= thresholds)
                all_scores.extend(body['scores'])
                all_decisions.extend(body['accepted'])
            scored = np.array(all_scores)
            np.testing.assert_equal(scored.shape, offline_scores[alias].shape)
            summary = compare_metrics(labels, -scored, subjects, artifact['models'][method],
                                      offline_report['results'][method])
            results[alias] = {'actual_frozen_method': method, 'subject_order_exact': True,
                'labels_exact': True, 'thresholds_exact': True, 'decision_flips': 0,
                'score_tolerance': {'atol': 1e-12, 'rtol': 1e-12},
                'max_absolute_score_error': float(np.max(abs(scored-offline_scores[alias]))),
                'strict_cmu_per_account_and_macro_metrics_exact': True, 'summary': summary}
        response = await client.get(f'/api/results/{DATASET}')
        response.raise_for_status()
        served = response.json()['validation']
        for alias, method in aliases.items():
            expected = results[alias]['summary']
            assert served[method]['aggregate_macro_user'] == {
                'eer': expected['macro_eer_diagnostic'], **expected['operating_points']['far_1pct']}
    assert original_hashes == {p.name: digest(p) for p in paths}
    result = {'dataset': DATASET, 'dev_samples': len(labels), 'enrolled_claims_per_sample': len(subjects),
              'scalar_scores_compared': len(labels) * len(subjects) * len(aliases),
              'scores_are_probabilities': False, 'score_direction': 'negative frozen distance; larger_is_genuine',
              'models': results, 'input_artifact_sha256_unchanged': original_hashes,
              'api_source_sha256_at_import': API_SOURCE_SHA256_AT_IMPORT,
              'api_source_sha256_at_completion': digest(API_SOURCE_PATH),
              'api_source_changed_during_run': API_SOURCE_SHA256_AT_IMPORT != digest(API_SOURCE_PATH),
              'test_access_denied': True, 'test_measurements_read': False,
              'network_listener_started': False, 'production_database_used': False,
              'elapsed_seconds': time.monotonic() - started}
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
