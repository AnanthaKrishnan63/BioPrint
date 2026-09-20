"""Guard against interpreting sealed measurements or fitting thresholds on dev."""
import csv
import numpy as np
import pytest
from eval.strict_cmu import load_partition, cmu_and_live_names, threshold_for, rates


def test_loader_does_not_parse_sealed_or_other_partition_values(tmp_path):
    columns, _ = cmu_and_live_names()
    path = tmp_path / 'poison.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'sessionIndex', 'rep'] + columns)
        writer.writerow(['a', 1, 1] + ['.1'] * len(columns))
        writer.writerow(['a', 7, 1] + ['SEALED_DO_NOT_PARSE'] * len(columns))
        writer.writerow(['a', 8, 1] + ['SEALED_DO_NOT_PARSE'] * len(columns))
    names, train = load_partition('train', path)
    assert train['a']['X'].shape == (1, 28)
    assert np.allclose(train['a']['X'], 100)
    assert load_partition('dev', path)[1] == {}
    with pytest.raises(ValueError, match='sealed'):
        load_partition('test', path)


def test_calibration_threshold_handles_ties_conservatively():
    genuine = np.array([0., 1., 2.])
    impostor = np.array([1., 1., 3., 4.])
    threshold = threshold_for(genuine, impostor, .25)
    assert rates(genuine, impostor, threshold)['far'] <= .25
    assert rates(genuine, impostor, threshold)['frr'] > 0
