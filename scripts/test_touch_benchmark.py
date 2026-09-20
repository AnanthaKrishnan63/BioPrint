"""Synthetic touch protocol and feature checks; no real measurements loaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import touch_benchmark as t


class TouchProtocolTests(unittest.TestCase):
    def test_reserved_participants_and_tasks_stay_sealed(self):
        self.assertEqual(t.role('user13', 'task1', 0), 'sealed_test')
        self.assertEqual(t.role('user01', 'task4', 0), 'sealed_test')
        self.assertEqual(t.role('user12', 'task3', 0), 'dev_probe')
        with self.assertRaisesRegex(ValueError, 'Sealed'):
            t.load(['sealed_test'])

    def test_train_loader_never_decodes_dev_or_sealed_features(self):
        with tempfile.TemporaryDirectory(dir=t.ROOT, prefix='.touch-test-') as folder:
            base = Path(folder)
            (base / 'keyboard_data.json').write_text(json.dumps({'keys_info': {
                'a': {'key_center_x': 10, 'key_center_y': 20, 'key_width': 10, 'key_height': 20}}}))
            header = b'participant_id,task_id,trial_id,timestamp_ms,ref_char,first_frame_touch_x,first_frame_touch_y\n'
            allowed = b''.join(f'user01,task1,0,{i*100},a,15,30\n'.encode() for i in range(5))
            excluded = b'user01,task3,0,\xff invalid dev\nuser01,task4,0,\xff sealed\nuser13,task1,0,\xff sealed\n'
            (base / 'touch_data.csv').write_bytes(header + allowed + excluded)
            with patch.object(t, 'DATA', base):
                rows, dropped = t.load(['training_enrollment_support'])
            self.assertEqual(len(rows), 1)
            self.assertEqual(dropped, [])
            np.testing.assert_allclose(rows[0]['x'][[0, 3]], [.5, .5])

    def test_gaps_never_include_nonpositive_or_long_pause(self):
        points = [[v, .1, -.1] for v in [0, 100, 100, 200, 300, 9000]]
        values = t.trial_features(points)
        np.testing.assert_allclose(values[6:], [100, 0, 100, 100])


if __name__ == '__main__':
    unittest.main()
