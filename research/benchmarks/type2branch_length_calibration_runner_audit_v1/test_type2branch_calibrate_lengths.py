import json
import pytest
import type2branch_calibrate_lengths as runner
from test_type2branch_length_completion import fixture


def test_missing_comparison_stops_before_arrays_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'OUT', tmp_path/'output')
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('Premature payload read'))
    with pytest.raises(FileNotFoundError): runner.main()
    assert not runner.OUT.exists()


def test_incomplete_arm_stops_before_arrays_and_output(tmp_path, monkeypatch):
    pair, arms = fixture(); arms['mixed']['updates'] = 500
    root = tmp_path/'research/benchmarks'
    directory = root/'type2branch_length_pair_v1'; directory.mkdir(parents=True)
    (directory/'report.json').write_text(json.dumps(pair))
    for name, report in arms.items():
        directory = root/f'type2branch_length_train_{name}_v1'; directory.mkdir()
        (directory/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'OUT', tmp_path/'output')
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('Premature payload read'))
    with pytest.raises(ValueError, match='complete600'): runner.main()
    assert not runner.OUT.exists()


@pytest.mark.parametrize('group', ['source_sha256', 'input_sha256'])
def test_changed_dependency_rejected(tmp_path, monkeypatch, group):
    target = tmp_path/'generated.txt'; target.write_text('before')
    manifest = {group: {'generated.txt': runner.sha(target)}}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    runner.verify_hashes(manifest)
    target.write_text('after')
    with pytest.raises(ValueError, match='Frozen dependency changed'):
        runner.verify_hashes(manifest)
