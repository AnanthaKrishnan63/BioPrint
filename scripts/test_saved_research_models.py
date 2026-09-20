"""Replay frozen learned models against their permitted dev feature matrices."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.research-deps'))
import joblib
import numpy as np
from device_benchmark import rates


class SavedResearchModelsTests(unittest.TestCase):
    def test_frozen_model_replays_match_reported_dev_rates(self):
        found = 0
        for family in ['device', 'delbot', 'cognitive']:
            directory = ROOT / 'research/benchmarks' / family
            if not (directory / 'schema.json').exists():
                self.fail(f'{family} artifacts missing; finish benchmark first')
            schema = json.loads((directory / 'schema.json').read_text())
            with np.load(directory / 'dev_features.npz', allow_pickle=False) as data:
                x, y = data['X'], data['y']
                self.assertEqual(x.shape[1], schema['feature_count'])
                self.assertTrue(np.isfinite(x).all())
                for name, config in schema['models'].items():
                    model = joblib.load(directory / config['artifact'])
                    scores = model.predict_proba(x)[:, 1]
                    for target, point in config['operating_points'].items():
                        actual = rates(y, scores, point['threshold_selected_on_training_calibration'])
                        with self.subTest(family=family, model=name, target=target):
                            self.assertEqual(actual, point['dev'])
                    found += 1
        self.assertEqual(found, 5)


if __name__ == '__main__':
    unittest.main()
