"""Integrity tests for split guards, adaptation and operating-point accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import device_benchmark as d
import numpy as np


class DeviceBenchmarkTests(unittest.TestCase):
    def test_reserved_loader_refuses_test(self):
        with self.assertRaisesRegex(ValueError, 'sealed'):
            d.load('test')

    def test_ties_cannot_exceed_target_far(self):
        labels = np.array([0, 0, 0, 0, 1])
        scores = np.array([.9, .9, .2, .1, .95])
        threshold = d.threshold_at_far(labels, scores, .25)
        self.assertGreater(threshold, .9)
        self.assertEqual(d.rates(labels, scores, threshold)['far'], 0.)

    def test_zero_far_threshold(self):
        labels = np.array([0, 0, 1])
        scores = np.array([.1, .2, .3])
        result = d.rates(labels, scores, d.threshold_at_far(labels, scores, 0.))
        self.assertEqual((result['far'], result['frr']), (0., 0.))

    def test_adapter_provides_available_signal(self):
        from engine import device
        row = {k: '' for k in d.FEATURES}
        row.update({'platformJS': 'Win32', 'userAgentHttp': 'Firefox/130',
                    'resolutionJS': '1920x1080x24', 'cookiesJS': 'yes'})
        env = d.env(row)
        result = device.check(env, [env])
        self.assertTrue(result.available)
        self.assertEqual(result.score, 0.)

    def test_prepare_never_decodes_reserved_features(self):
        test_id = next(str(i) for i in range(100) if d.partition(str(i)) == 'test')
        train_id = next(str(i) for i in range(100) if d.partition(str(i)) == 'train')
        values = ['1', train_id] + ['x'] * (len(d.COLUMNS) - 2)
        permitted = ('(' + ','.join("'" + v + "'" if i else v for i, v in enumerate(values)) + ');\n').encode()
        # Valid metadata followed by invalid UTF8: any test decoding would fail.
        sealed = f"(2,'{test_id}',".encode() + b'\xff\xfe never parse this);\n'
        with tempfile.TemporaryDirectory(dir=d.ROOT, prefix='.device-test-') as directory:
            with patch.object(d, 'DATA', Path(directory)), patch.object(d, 'rows', lambda: iter([permitted, sealed])):
                d.prepare()
            self.assertFalse((Path(directory) / 'test.jsonl').exists())
            stored = json.loads((Path(directory) / 'train.jsonl').read_text())
            self.assertEqual(stored['id'], train_id)
            self.assertNotIn('addressHttp', stored)


if __name__ == '__main__':
    unittest.main()
