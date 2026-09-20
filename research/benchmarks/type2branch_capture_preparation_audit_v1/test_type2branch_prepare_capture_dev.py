import pytest
import type2branch_prepare_capture_dev as prepare


def test_missing_pair_prevents_dev_data_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare, 'OUT', tmp_path/'output')
    monkeypatch.setattr(prepare, 'select_dev_rows', lambda *a: pytest.fail('Premature DEV read'))
    monkeypatch.setattr(prepare.np, 'load', lambda *a, **k: pytest.fail('Premature array read'))
    with pytest.raises(FileNotFoundError): prepare.main()
    assert not prepare.OUT.exists()
