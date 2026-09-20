"""Filesystem-only budget guards; no dataset observations are read."""
import pytest
from dataset_budget import apparent_bytes, source_document_files


def test_internal_fixture_alias_counts_link_without_duplicate_payload(tmp_path):
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    payload = fixture / 'synthetic.bin'
    payload.write_bytes(b'x' * 123)
    alias = tmp_path / 'current'
    alias.symlink_to(fixture, target_is_directory=True)
    expected = tmp_path.stat().st_size + fixture.stat().st_size + 123 + alias.lstat().st_size
    assert apparent_bytes(tmp_path, allow_internal_aliases=True) == expected


def test_temporary_alias_cannot_escape_scanned_directory(tmp_path):
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'synthetic')
    (fixture / 'alias').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        apparent_bytes(fixture, allow_internal_aliases=True)


def test_dataset_symlink_still_requires_explicit_audit(tmp_path):
    (tmp_path / 'synthetic.bin').write_bytes(b'x')
    (tmp_path / 'alias').symlink_to(tmp_path / 'synthetic.bin')
    with pytest.raises(ValueError, match='symlink'):
        apparent_bytes(tmp_path)


def test_source_documents_count_all_formats_without_double_counting(tmp_path, monkeypatch):
    pdf = tmp_path / 'hmog_schema.pdf'
    receipt = tmp_path / 'hmog_receipt.json'
    array = tmp_path / 'synthetic.npy'
    pdf.write_bytes(b'pdf'); receipt.write_bytes(b'{}'); array.write_bytes(b'array')
    monkeypatch.setattr(type(pdf), 'read_bytes', lambda _: pytest.fail('Observation contents must not be read'))
    assert source_document_files(tmp_path, [array]) == {pdf: 3, receipt: 2}


def test_source_document_symlinks_require_explicit_audit(tmp_path):
    (tmp_path / 'schema.pdf').write_bytes(b'doc')
    (tmp_path / 'alias.pdf').symlink_to(tmp_path / 'schema.pdf')
    with pytest.raises(ValueError, match='symlink'):
        source_document_files(tmp_path)
