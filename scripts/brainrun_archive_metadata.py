"""Read only BrainRun ZIP directory metadata through bounded HTTP ranges."""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/brainrun/archive_metadata_v1'
SIZE=265000157
URL='https://zenodo.org/api/records/2598135/files/gestures_devices_users_games_data.zip/content'


def fetch(start,end):
    if not 0<=start<=end<SIZE or end-start+1>2_000_000:
        raise ValueError('Metadata range exceeds bound')
    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=45) as response:
        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
            raise ValueError('Refusing nonexact range response before payload read')
        payload=response.read(end-start+2)
    if len(payload)!=end-start+1:raise ValueError('Unexpected range length')
    return payload


def parse_directory(payload,count):
    entries=[];pos=0
    while pos<len(payload):
        if len(payload)-pos<46:raise ValueError('Truncated central header')
        h=struct.unpack_from('<4s6H3L5H2L',payload,pos)
        if h[0]!=b'PK\x01\x02':raise ValueError('Not a central directory')
        n,e,c=h[10:13];end=pos+46+n+e+c
        if end>len(payload):raise ValueError('Truncated central entry')
        name=payload[pos+46:pos+46+n].decode('utf-8' if h[3]&0x800 else 'cp437')
        if h[8]==0xffffffff or h[9]==0xffffffff or h[16]==0xffffffff:
            raise ValueError('ZIP64 requires separate explicit audit')
        entries.append({'name':name,'compressed_bytes':h[8],'uncompressed_bytes':h[9],
            'method':h[4],'flags':h[3],'crc32':h[7],'local_header_offset':h[16]})
        pos=end
    if len(entries)!=count:raise ValueError('Directory count mismatch')
    return entries


def main():
    OUT.mkdir(exist_ok=False)
    plan={'url':URL,'expected_archive_bytes':SIZE,'scope':'ZIP directory only; no member decompression or observation parsing',
        'maximum_directory_bytes':2_000_000,'first_range':[SIZE-22,SIZE-1],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    try:
        tail=fetch(SIZE-22,SIZE-1)
        h=struct.unpack('<4s4H2LH',tail)
        if h[0]!=b'PK\x05\x06' or h[1]!=0 or h[2]!=0 or h[3]!=h[4] or h[7]!=0:
            raise ValueError('Only single-disk comment-free ZIP metadata supported')
        count,size,offset=h[4:7]
        if not count or size>2_000_000 or offset+size!=SIZE-22:
            raise ValueError('Unexpected directory extent')
        directory=fetch(offset,offset+size-1)
        entries=parse_directory(directory,count)
        report={'status':'archive_directory_verified','source_url':URL,'archive_bytes':SIZE,
            'directory_sha256':hashlib.sha256(directory).hexdigest(),'directory_bytes':size,'entries':entries,
            'total_uncompressed_bytes':sum(r['uncompressed_bytes'] for r in entries),
            'observations_decoded':False,'member_payloads_requested':False}
        (OUT/'directory.bin').write_bytes(directory)
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception as error:
        (OUT/'failure.json').write_text(json.dumps({'status':'metadata_acquisition_failed','error_type':type(error).__name__,
            'error':str(error),'observations_decoded':False},indent=2)+'\n')
        raise


if __name__=='__main__':main()
