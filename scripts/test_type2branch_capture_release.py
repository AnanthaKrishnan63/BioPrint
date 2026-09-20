import json
import pytest
import type2branch_capture_release as release
import type2branch_build_capture_release as builder


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();directory=tmp_path/'release';directory.mkdir()
    monkeypatch.setattr(release,'ROOT',root)
    for name in release.SOURCES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'generated source')
    for name in release.FILES:(directory/name).write_bytes(b'generated artifact')
    manifest={'version':1,'protocol':'capture','split':'dev',
        'files':{n:release.digest((directory/n).read_bytes()) for n in release.FILES},
        'sources':{n:release.digest((root/n).read_bytes()) for n in release.SOURCES}}
    raw=json.dumps(manifest).encode();(directory/'manifest.json').write_bytes(raw)
    return root,directory,release.digest(raw)


def test_generated_inventory_verified(tmp_path,monkeypatch):
    _,directory,pin=fixture(tmp_path,monkeypatch)
    assert release.verify_release(pin,directory)['protocol']=='capture'


@pytest.mark.parametrize('fault',['artifact','source','pin','manifest'])
def test_tampering_rejected(tmp_path,monkeypatch,fault):
    root,directory,pin=fixture(tmp_path,monkeypatch)
    if fault=='artifact':(directory/'profiles.npz').write_bytes(b'changed')
    if fault=='source':(root/'scripts/type2branch_capture_worker.py').write_bytes(b'changed')
    if fault=='pin':pin='0'*64
    if fault=='manifest':(directory/'manifest.json').write_text('{}')
    with pytest.raises(ValueError):release.verify_release(pin,directory)


def test_release_builder_requires_completed_dev_before_output(tmp_path,monkeypatch):
    monkeypatch.setattr(builder,'ROOT',tmp_path);monkeypatch.setattr(builder,'RELEASE',tmp_path/'output')
    monkeypatch.setattr(builder.np,'load',lambda *a,**k:pytest.fail('Premature payload read'))
    with pytest.raises(FileNotFoundError):builder.main()
    assert not builder.RELEASE.exists()
