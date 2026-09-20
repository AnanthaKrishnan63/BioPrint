import json
import pytest
import type2branch_release as release


def fixture(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir()
    directory=tmp_path/'release';directory.mkdir()
    for name in release.SOURCES:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('generated source')
    for name in release.FILES:(directory/name).write_text('generated artifact')
    monkeypatch.setattr(release,'ROOT',source)
    manifest={'version':1,'split':'dev',
        'files':{name:release.digest((directory/name).read_bytes()) for name in release.FILES},
        'sources':{name:release.digest((source/name).read_bytes()) for name in release.SOURCES}}
    raw=json.dumps(manifest).encode();(directory/'manifest.json').write_bytes(raw)
    return directory,source,release.digest(raw)


def test_complete_generated_release(tmp_path,monkeypatch):
    directory,source,pin=fixture(tmp_path,monkeypatch)
    assert release.verify_release(pin,directory)['split']=='dev'


@pytest.mark.parametrize('fault',['manifest','checkpoint','source','pin'])
def test_corruption_rejected_before_deserialization(tmp_path,monkeypatch,fault):
    directory,source,pin=fixture(tmp_path,monkeypatch)
    if fault=='manifest':(directory/'manifest.json').write_text('{}')
    if fault=='checkpoint':(directory/'encoder.index').write_text('changed')
    if fault=='source':(source/'scripts/type2branch_worker.py').write_text('changed')
    if fault=='pin':pin='not a hash'
    with pytest.raises(ValueError):release.verify_release(pin,directory)
