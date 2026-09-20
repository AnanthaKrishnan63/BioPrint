"""Synthetic protocol checks for the SapiMouse architecture port."""
import unittest
import zlib
from unittest.mock import patch
from engine.pointer_sequence import displacement_blocks
import numpy as np
import torch
from pointer_sapimouse_benchmark import FCN, aggregate, load

class SapiMouseTests(unittest.TestCase):
    def test_sealed_user_read_fails(self):
        with self.assertRaisesRegex(ValueError, 'Refusing test'):
            load([{'split': 'test_sealed'}])

    def test_fcn_architecture(self):
        model = FCN(60).eval()
        with torch.no_grad():
            value = torch.zeros(2, 2, 128)
            self.assertEqual(tuple(model.embed(value).shape), (2, 128))
            self.assertEqual(tuple(model(value).shape), (2, 60))
        convs = [x for x in model.modules() if isinstance(x, torch.nn.Conv1d)]
        self.assertEqual([x.kernel_size[0] for x in convs], [8, 5, 3])
        self.assertEqual([x.out_channels for x in convs], [128, 256, 128])

    def test_training_serving_preprocessing_parity(self):
        time = np.arange(257)
        xy = np.column_stack([time ** 1.2, np.sin(time / 5)])
        lines = ['client timestamp,button,state,x,y']
        lines.extend(f'{i * 10},NoButton,Move,{x},{y}' for i, (x, y) in enumerate(xy))
        content = ('\n'.join(lines) + '\n').encode()
        entry = {'split': 'train', 'path': 'synthetic/user1/session_3min.csv', 'crc': zlib.crc32(content)}
        with patch('pathlib.Path.read_bytes', return_value=content):
            training_blocks, _, _, _ = load([entry])
        np.testing.assert_allclose(training_blocks, displacement_blocks(xy), atol=1e-7)

    def test_aggregation_never_crosses_sessions(self):
        s = np.array([[1., 2., 3., 100., 200., 300.]])
        grouped, labels, sessions = aggregate(s, np.array(['a'] * 3 + ['b'] * 3), np.array(['s1'] * 3 + ['s2'] * 3), 2)
        np.testing.assert_array_equal(grouped, [[1.5, 150.]])
        np.testing.assert_array_equal(labels, ['a', 'b'])
        np.testing.assert_array_equal(sessions, ['s1', 's2'])

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
