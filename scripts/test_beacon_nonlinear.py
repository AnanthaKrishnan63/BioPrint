"""Frozen behavior-only models must ignore advisory context; generated inputs."""
import unittest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch
from beacon_nonlinear_benchmark import OUT, FIT, SELECT, CAL, check_protocol
import joblib
import numpy as np
import httpx
import research_modalities as routes
from research_api import app
from fastapi import HTTPException


class NonlinearFusionTests(unittest.TestCase):
    def test_training_cohorts_are_disjoint_and_protocol_unchanged(self):
        self.assertFalse(set(FIT) & set(SELECT))
        self.assertFalse(set(FIT) & set(CAL))
        self.assertFalse(set(SELECT) & set(CAL))
        self.assertEqual(check_protocol()['test_policy'],
            'Original metadata split and role ledger; no reserved measurements loaded.')

    def test_behavior_models_cannot_use_context(self):
        x = np.random.default_rng(11).uniform(0, 1, (16, 33))
        changed = x.copy()
        changed[:, 32] = 7
        for name in ['behavior_fusion_selected', 'behavior_fusion_reference_lr']:
            model = joblib.load(OUT / (name + '.joblib'))
            np.testing.assert_array_equal(model.predict_proba(x), model.predict_proba(changed))

    def test_exported_schema_uses_corrected_v2_thresholds(self):
        corrected = json.loads((OUT / 'frozen_v2.json').read_text())
        original = json.loads((OUT / 'frozen.json').read_text())
        routes.tabular_schema.cache_clear()
        schema = routes.tabular_schema('beacon_nonlinear')
        self.assertEqual(set(schema['models']), set(corrected['models']))
        for name, spec in schema['models'].items():
            for target, operating in spec['operating_points'].items():
                self.assertEqual(operating['threshold_selected_on_training_calibration'],
                                 corrected['models'][name]['thresholds'][target])
            self.assertNotEqual(corrected['models'][name]['thresholds']['0.01'],
                                original['models'][name]['thresholds']['0.01'])

    def test_asgi_returns_v2_threshold_without_recording_access(self):
        corrected = json.loads((OUT / 'frozen_v2.json').read_text())
        async def run():
            transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
            async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
                with patch.object(routes.np, 'load', side_effect=AssertionError('Recording read')):
                    sealed = await client.get('/api/features/beacon_nonlinear/test/samples')
                    self.assertEqual(sealed.status_code, 403)
                    for name, spec in corrected['models'].items():
                        response = await client.post(f'/api/features/beacon_nonlinear/models/{name}/score',
                                                     json={'features': [[0.0] * 33]})
                        self.assertEqual(response.status_code, 200)
                        body = response.json()
                        self.assertEqual(body['threshold'], spec['thresholds']['0.01'])
                        self.assertEqual(body['accepted'], [body['scores'][0] >= body['threshold']])
        asyncio.run(run())

    def test_tampered_export_fails_before_deserialization_even_with_cached_schema(self):
        original_read = Path.read_bytes
        paths = ['schema.json', 'behavior_fusion_selected.joblib', 'frozen_v2.json',
                 'preregistered.json', 'export_integrity.json']
        for name in paths:
            with self.subTest(tampered=name):
                routes.tabular_schema.cache_clear()
                routes.tabular_model.cache_clear()
                routes.tabular_schema('beacon_nonlinear')
                target = OUT / name
                def changed_read(path):
                    raw = original_read(path)
                    return raw + b'\nTAMPERED' if path == target else raw
                with patch.object(Path, 'read_bytes', changed_read), \
                     patch('joblib.load', side_effect=AssertionError('Unverified deserialization')):
                    with self.assertRaises(HTTPException) as caught:
                        routes.tabular_model('beacon_nonlinear', 'behavior_fusion_selected')
                    self.assertEqual(caught.exception.status_code, 503)
        routes.tabular_schema.cache_clear()
        routes.tabular_model.cache_clear()


if __name__ == '__main__':
    unittest.main()
