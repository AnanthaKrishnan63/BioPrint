"""Acquire the bounded compressed BrainRun archive without decoding records."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/brainrun'
OUT=DATA/'acquisition_v1'
SIZE=265000157
MD5='a098d5a4028f38a97841c40470ca73e8'
URL='https://zenodo.org/api/records/2598135/files/gestures_devices_users_games_data.zip/content'


def main():
    metadata_path=DATA/'record_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    source=next(r for r in metadata['files'] if r['key']=='gestures_devices_users_games_data.zip')
    if source['size']!=SIZE or source['checksum']!='md5:'+MD5:raise ValueError('Published source changed')
    directory_path=DATA/'archive_metadata_v1/report.json';directory=json.loads(directory_path.read_text())
    if directory['status']!='archive_directory_verified' or directory['archive_bytes']!=SIZE:
        raise ValueError('Verified archive directory required')
    current=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    if current+SIZE+10_000_000>=500_000_000:raise ValueError('Source budget insufficient')
    # The independently audited full corpus was1.212GB; this acquisition adds265MB.
    budget_path=ROOT/'research/benchmarks/dataset_budget_paired_type2branch_complete.json'
    budget=json.loads(budget_path.read_text())
    if not budget['within_budget'] or budget['total_bytes']+SIZE+10_000_000>=5_000_000_000:
        raise ValueError('Overall budget insufficient')
    OUT.mkdir(exist_ok=False)
    plan={'url':URL,'expected_bytes':SIZE,'published_md5':MD5,
        'scope':'Compressed archive only; no extraction, member decompression or observation decoding',
        'maximum_source_bytes':500_000_000,'reserved_derived_bytes':10_000_000,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),metadata_path,directory_path,budget_path]}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    dest=OUT/'gestures_devices_users_games_data.zip';partial=dest.with_suffix('.zip.partial')
    received=0;md5=hashlib.md5();sha=hashlib.sha256();started=time.perf_counter()
    try:
        # Explicit full range avoids a cached directory-only response at the same URL.
        req=urllib.request.Request(URL,headers={'Range':f'bytes=0-{SIZE-1}','Cache-Control':'no-cache'})
        with urllib.request.urlopen(req,timeout=60) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes 0-{SIZE-1}/{SIZE}':
                raise ValueError('Exact full-archive range required before reading payload')
            with partial.open('xb') as f:
                while received<SIZE:
                    chunk=response.read(min(1_048_576,SIZE-received))
                    if not chunk:raise ValueError('Truncated acquisition')
                    f.write(chunk);md5.update(chunk);sha.update(chunk);received+=len(chunk)
                if response.read(1):raise ValueError('Oversized acquisition')
        if md5.hexdigest()!=MD5:raise ValueError('Published checksum mismatch')
        with zipfile.ZipFile(partial) as z:
            actual=[{'name':i.filename,'compressed_bytes':i.compress_size,'uncompressed_bytes':i.file_size,
                'method':i.compress_type,'flags':i.flag_bits,'crc32':i.CRC,'local_header_offset':i.header_offset} for i in z.infolist()]
        if actual!=directory['entries']:raise ValueError('Archive directory differs from preregistered metadata')
        partial.rename(dest)
        report={'status':'compressed_archive_acquired','bytes':received,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),
            'seconds':time.perf_counter()-started,'member_directory_verified':True,
            'members_decompressed':False,'observations_decoded':False,'archive':str(dest.relative_to(ROOT))}
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    except Exception as error:
        (OUT/'failure.json').write_text(json.dumps({'status':'acquisition_failed','error':str(error),
            'error_type':type(error).__name__,'bytes_received':received,'partial_preserved':partial.exists(),
            'observations_decoded':False},indent=2)+'\n')
        raise


if __name__=='__main__':main()
