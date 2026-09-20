"""Synthetic API boundary checks; no research recordings are read."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import httpx
import numpy as np
from research_api import app
import research_modalities as routes


def request(method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


class ModalityReviewTests(unittest.TestCase):
    def test_sealed_routes_do_not_open_artifacts(self):
        with patch.object(routes.np, 'load', side_effect=AssertionError('Dataset read')):
            for path in ['/api/paired/beacon/test/samples', '/api/features/device/test/samples',
                         '/api/features/cognitive/test/samples', '/api/features/delbot/test/samples']:
                self.assertEqual(request('GET', path).status_code, 403)

    def test_unrepresentable_input_is_rejected_before_model_loading(self):
        schema = {'feature_count': 2}
        with patch.object(routes, 'tabular_schema', return_value=schema), \
             patch.object(routes, 'tabular_model', side_effect=AssertionError('Model loaded')):
            for value in [1e100, -1e100, 'NaN', 'Infinity']:
                self.assertEqual(request('POST', '/api/features/device/models/example/score',
                    json={'features': [[value, 0]]}).status_code, 422)
        with patch.object(routes, 'configuration', side_effect=AssertionError('Config loaded')):
            self.assertEqual(request('POST', '/api/paired/beacon/models/equal_behavior/score',
                json={'features': [[1e308] * 33]}).status_code, 422)

    def test_equal_behavior_direction_and_inclusive_threshold(self):
        with patch.object(routes, 'configuration', return_value={'baseline_thresholds': {'far_1pct': .5}}):
            body = request('POST', '/api/paired/beacon/models/equal_behavior/score',
                json={'features': [[v] * 32 + [100] for v in [.25, .5, .75]]}).json()
        self.assertEqual(body['scores'], [-.25, -.5, -.75])
        self.assertEqual(body['threshold'], -.5)
        self.assertEqual(body['accepted'], [True, True, False])

    def test_no_silent_clamping_of_representable_values(self):
        x = routes.feature_matrix(routes.PairBatch(features=[[1e20, -1e20]]), 2)
        np.testing.assert_array_equal(x, [[1e20, -1e20]])

    def test_validated_forest_loader_changes_runtime_threads_only(self):
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=160, max_depth=12, n_jobs=2)
        before = forest.get_params().copy()
        with patch.object(routes, 'tabular_schema', return_value={
            'models': {'fpstalker_inspired_random_forest': {'artifact': 'generated.joblib'}}}), \
             patch.object(routes, 'artifact_directory', return_value=ROOT), \
             patch('joblib.load', return_value=forest):
            routes.tabular_model.cache_clear()
            loaded = routes.tabular_model('device', 'fpstalker_inspired_random_forest')
            after = loaded.get_params()
        routes.tabular_model.cache_clear()
        self.assertEqual(after.pop('n_jobs'), 1)
        before.pop('n_jobs')
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
