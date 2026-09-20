"""Bounded HMOG acquisition; never reads inner ZIP measurement members."""
import argparse
import hashlib
import json
import pathlib
import struct
import time
import urllib.request
import zipfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
META = ROOT / 'datasets/joint_feasibility_metadata'
OUT = ROOT / 'datasets/hmog'
TOTAL = 6132356276
LIMIT = 500000000
SOURCE = 'https://wm1693.app.box.com/index.php?rm=box_download_shared_file&shared_name=jkjodaw2scunua7b7qaxt3ker5w9lwt7&file_id=f_770692248712'

def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def disk_bytes():
    return sum(p.stat().st_size for base in (META, OUT) for p in base.rglob('*') if p.is_file())

def members():
    data = (META / 'hmog_zip_tail.bin').read_bytes()
    doc = json.loads((META / 'hmog_zip_member_metadata.json').read_text())
    pos = doc['zip64_eocd']['central_directory_offset'] - (TOTAL - len(data))
    checksums = {}
    while data[pos:pos + 4] == b'PK\x01\x02':
        row = struct.unpack_from('<4s6H3L5H2L', data, pos)
        n, e, c = row[10:13]
        name = data[pos + 46:pos + 46 + n].decode()
        checksums[name] = {'crc32': row[7], 'flags': row[3]}
        pos += 46 + n + e + c
    result = [dict(m, **checksums[m['name']]) for m in doc['members'] if m['name'].endswith('.zip')]
    assert len(result) == 100
    return sorted(result, key=lambda m: (m['uncompressed_bytes'], m['name']))

def prepare():
    OUT.mkdir(exist_ok=True)
    ms = members()
    cohort = ms[:12]
    order = sorted(cohort, key=lambda m: hashlib.sha256(('20260920:' + pathlib.Path(m['name']).stem).encode()).hexdigest())
    roles = ['fit'] * 4 + ['selection'] * 2 + ['calibration'] * 2 + ['dev'] * 4
    for m, role in zip(order, roles):
        m['identity_role'] = role
        m['measurement_state'] = 'sealed_pending_session_role_preregistration'
    doc = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'source_url': SOURCE,
           'source_file_id': '770692248712', 'source_version': '822468323512', 'source_bytes': TOTAL,
           'source_sha1': 'ff7970cdd4a16bc918b5af941515363f806f9b1f', 'metadata_sha256': sha(META / 'hmog_zip_member_metadata.json'),
           'selection': '12 smallest nested subject ZIPs by uncompressed byte count, name tie break; completeness/recording-length selection bias',
           'partition_rule': 'SHA256 UTF-8 string 20260920:<subject ID>, ascending digest: 4 fit, 2 selection, 2 calibration, 4 dev',
           'cohort': order, 'sealed_test_never_fetch': ms[12:16], 'unused_sealed_never_fetch': ms[16:],
           'raw_inner_zip_bytes': sum(m['uncompressed_bytes'] for m in cohort), 'disk_ceiling_exclusive_bytes': LIMIT,
           'scope': 'Acquisition and inner ZIP directory metadata only. No measurement member read/decode; all session support/probe roles pending.',
           'license': 'Official HMOG noncommercial education/research terms; no redistribution.'}
    assert len({m['name'] for m in doc['cohort']}) == 12
    assert len(doc['sealed_test_never_fetch']) == 4 and len(doc['unused_sealed_never_fetch']) == 84
    assert disk_bytes() + doc['raw_inner_zip_bytes'] + 2000000 < LIMIT
    dump(OUT / 'preregistered_acquisition.json', doc)
    print('Preregistered', doc['raw_inner_zip_bytes'], 'raw ZIP bytes', flush=True)

def response(start, end):
    url = SOURCE + '&_=' + str(time.time_ns())
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'})
    r = urllib.request.urlopen(req, timeout=60)
    expected = f'bytes {start}-{end}/{TOTAL}'
    if r.status != 206 or r.headers.get('Content-Range') != expected or int(r.headers.get('Content-Length', '-1')) != end - start + 1:
        r.close()
        raise RuntimeError('Refusing nonexact HTTP range response')
    return r

def fetch(m):
    name = pathlib.Path(m['name']).name
    dest = OUT / 'raw' / name
    receipt = OUT / 'receipts' / (name + '.json')
    partial = dest.with_suffix('.zip.partial')
    if dest.exists() or receipt.exists() or partial.exists():
        raise FileExistsError(f'Refusing overwrite: {dest}')
    if disk_bytes() + m['uncompressed_bytes'] + 2000000 >= LIMIT:
        raise RuntimeError('Disk ceiling would be exceeded')
    start = m['local_header_offset']
    with response(start, start + 29) as r:
        header = r.read(30)
    row = struct.unpack('<4s5H3L2H', header)
    assert row[0] == b'PK\x03\x04' and row[3] == 8 and not (row[2] & 1)
    n, e = row[-2:]
    with response(start + 30, start + 29 + n + e) as r:
        extra = r.read(n + e)
    assert extra[:n].decode() == m['name']
    payload_start = start + 30 + n + e
    payload_end = payload_start + m['compressed_bytes'] - 1
    began = time.time()
    written = received = crc = 0
    h = hashlib.sha256()
    decoder = zlib.decompressobj(-15)
    with response(payload_start, payload_end) as r, partial.open('xb') as f:
        while received < m['compressed_bytes']:
            b = r.read(min(262144, m['compressed_bytes'] - received))
            if not b:
                raise RuntimeError('Truncated range')
            received += len(b)
            pending = b
            while pending:
                output = decoder.decompress(pending, min(1048576, m['uncompressed_bytes'] - written + 1))
                pending = decoder.unconsumed_tail
                if written + len(output) > m['uncompressed_bytes'] or disk_bytes() + len(output) >= LIMIT:
                    raise RuntimeError('Inflation/disk bound exceeded')
                f.write(output)
                written += len(output)
                crc = zlib.crc32(output, crc)
                h.update(output)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    assert received == m['compressed_bytes'] and written == m['uncompressed_bytes'] and crc == m['crc32']
    partial.rename(dest)
    # infolist reads directory metadata, never member data.
    with zipfile.ZipFile(dest) as z:
        index = [{'name': i.filename, 'compressed_bytes': i.compress_size, 'uncompressed_bytes': i.file_size, 'crc32': i.CRC} for i in z.infolist()]
    dump(receipt, {'subject': dest.stem, 'identity_role': m['identity_role'], 'range': [payload_start, payload_end],
                   'http_status': 206, 'content_range': f'bytes {payload_start}-{payload_end}/{TOTAL}',
                   'compressed_received': received, 'raw_inner_zip_bytes': written, 'crc32_verified': crc,
                   'sha256': h.hexdigest(), 'elapsed_seconds': time.time() - began, 'inner_member_metadata': index,
                   'measurements_decoded': False})
    print('Completed', dest.stem, written, 'bytes', round(time.time() - began, 1), 'seconds', flush=True)

def acquire():
    p = OUT / 'preregistered_acquisition.json'
    doc = json.loads(p.read_text())
    assert doc['source_bytes'] == TOTAL and doc['metadata_sha256'] == sha(META / 'hmog_zip_member_metadata.json')
    for d in ('raw', 'receipts'):
        (OUT / d).mkdir(exist_ok=True)
    forbidden = {m['name'] for m in doc['sealed_test_never_fetch'] + doc['unused_sealed_never_fetch']}
    assert not forbidden.intersection(m['name'] for m in doc['cohort']) and len(doc['cohort']) == 12
    for m in doc['cohort']:
        fetch(m)
    dump(OUT / 'acquisition_complete.json', {'preregistration_sha256': sha(p), 'subjects': 12, 'disk_bytes_including_discovery_metadata': disk_bytes(), 'measurements_decoded': False})

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('action', choices=['prepare', 'acquire']); args = a.parse_args()
    if args.action == 'prepare': prepare()
    else: acquire()
