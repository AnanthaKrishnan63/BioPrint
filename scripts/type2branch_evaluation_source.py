"""Bounded acquisition of evaluation source from the already pinned tree."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    ref=ROOT/'research/benchmarks/references/type2branch'
    out=ROOT/'research/benchmarks/references/type2branch_evaluation'
    out.mkdir(exist_ok=False)
    tree=json.loads((ref/'tree.json').read_text())
    entry=next(x for x in tree['tree'] if x['path']=='evaluate.py')
    url=f"https://raw.githubusercontent.com/lsia/tifs-type2branch/{tree['sha']}/evaluate.py"
    request=urllib.request.Request(url,headers={'User-Agent':'bioprint-research-source-audit'})
    with urllib.request.urlopen(request,timeout=25) as response:
        body=response.read(2_000_001)
    if len(body)>2_000_000:raise ValueError('Source exceeds bound')
    if len(body)!=entry['size']:raise ValueError('Publisher size mismatch')
    blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if blob!=entry['sha']:raise ValueError('Publisher blob hash mismatch')
    (out/'evaluate.py').write_bytes(body)
    (out/'LICENSE').write_bytes((ref/'LICENSE').read_bytes())
    receipt={'url':url,'commit':tree['sha'],'git_blob_sha1':blob,'bytes':len(body),
             'sha256':hashlib.sha256(body).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
