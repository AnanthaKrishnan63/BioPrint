"""Snapshot official legacy Random source and modern seeded compatibility vectors."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/references/type2branch_random'


def main():
    OUT.mkdir(exist_ok=False)
    receipts=[]
    def fetch(url,name):
        with urllib.request.urlopen(url,timeout=25) as response:body=response.read(2_000_001)
        if len(body)>2_000_000:raise ValueError('Source bound exceeded')
        (OUT/name).write_bytes(body)
        receipts.append({'url':url,'file':name,'sha256':hashlib.sha256(body).hexdigest()})
        return body
    for repo,name,path,license_path in [
        ('microsoft/referencesource','legacy','mscorlib/system/random.cs','LICENSE.txt'),
        ('dotnet/runtime','tests','src/libraries/System.Runtime/tests/System.Runtime.Extensions.Tests/System/Random.cs','LICENSE.TXT')]:
        commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main',name+'_commit.json'))['sha']
        prefix='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'
        fetch(prefix+path,name+'.cs')
        fetch(prefix+license_path,name+'_LICENSE.txt')
    (OUT/'receipts.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(receipts))


if __name__=='__main__':main()
