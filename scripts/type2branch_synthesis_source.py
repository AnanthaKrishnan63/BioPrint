"""Acquire only publisher source metadata/docs, never example recordings."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/references/type2branch_synthesis'
REPO = 'SoftwareImpacts/SIMPAC-2022-276'


def fetch(url, name):
    with urllib.request.urlopen(url, timeout=25) as response:
        body = response.read(2_000_001)
        if len(body) > 2_000_000: raise ValueError('Metadata/source bound exceeded')
    with (OUT / name).open('xb') as stream: stream.write(body)
    return body


def main():
    OUT.mkdir(exist_ok=False)
    tree = json.loads(fetch('https://api.github.com/repos/' + REPO + '/git/trees/main?recursive=1', 'tree.json'))
    if tree.get('truncated'): raise ValueError('Incomplete tree')
    for name in ['README.md', 'LICENSE']:
        fetch('https://raw.githubusercontent.com/' + REPO + '/' + tree['sha'] + '/' + name, name)
    source_files = [r for r in tree['tree'] if r['type'] == 'blob' and
                    r['path'].startswith('KSD-SLD/') and r['path'].endswith(('.cs', '.csproj'))
                    and '/obj/' not in r['path'] and '/bin/' not in r['path']]
    (OUT / 'manifest.json').write_text(json.dumps({'repository': REPO, 'tree_sha': tree['sha'],
        'source_inventory': source_files, 'checksums': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
        for name in ['tree.json', 'README.md', 'LICENSE']}, 'example_data_downloaded': False}, indent=2))
    print(json.dumps({'tree_sha': tree['sha'], 'source_inventory': [(r['path'], r['size']) for r in source_files]}))


def sources():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['checksums'].items():
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Metadata changed')
    extra = ['Program.cs', 'FiniteContexts/Profiles/Profile.cs',
             'FiniteContexts/Profiles/FiniteContextsConfiguration.cs',
             'FiniteContexts/Models/ModelFeeder.cs',
             'FiniteContexts/Models/AvgStdev/AvgStdevModelLinear.cs',
             'FiniteContexts/Models/Histogram/HistogramModel.cs',
             'FiniteContexts/ContextSelection/LongestAvailableContextSelector.cs',
             'Util/RNG.cs', 'Datasets/Readers/CsvDatasetReader.cs']
    rows = [r for r in manifest['source_inventory'] if '/Synthesizer/' in r['path']
            or r['path'] in ['KSD-SLD/' + p for p in extra]]
    if sum(r['size'] for r in rows) > 500_000: raise ValueError('Source budget exceeded')
    target = OUT / 'source'; target.mkdir(exist_ok=False)
    def download(row):
        name = row['path']
        if '..' in Path(name).parts: raise ValueError('Invalid source path')
        dest = target / name; dest.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://raw.githubusercontent.com/' + REPO + '/' + manifest['tree_sha'] + '/' + name
        body = fetch(url, str(dest.relative_to(OUT)))
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if len(body) != row['size'] or blob != row['sha']: raise ValueError('Publisher source mismatch')
        return {'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(download, rows))
    (OUT / 'source_receipts.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps({'source_files': len(receipts), 'bytes': sum(r['bytes'] for r in receipts),
                      'measurements_downloaded': False}))


def configuration(cleanup=False, context=False, session=False):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['checksums'].items():
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Metadata changed')
    tree = json.loads((OUT / 'tree.json').read_text())
    wanted = ({'KSD-SLD/FiniteContexts/Partitions/CleanFTs.cs',
               'KSD-SLD/FiniteContexts/Partitions/ThresholdPartitioner.cs'} if cleanup
              else {'KSD-SLD/App.config', 'KSD-SLD/packages.config'})
    if context:
        wanted = {'KSD-SLD/FiniteContexts/' + p for p in [
            'ContextSelection/ContextSelector.cs', 'ModelStorages/MemoryStorage.cs',
            'ModelStorages/ModelStorage.cs', 'Models/AvgStdev/AvgStdevModel.cs',
            'Models/StandardModelFactory.cs', 'PatternVector/Builder.cs']}
    if session:
        wanted = {'KSD-SLD/Datasets/Session.cs'}
    rows = [r for r in tree['tree'] if r['path'] in wanted and r['type'] == 'blob']
    if {r['path'] for r in rows} != wanted:
        raise ValueError('Configuration inventory incomplete')
    folder = 'session' if session else ('context' if context else ('cleanup' if cleanup else 'configuration'))
    target = OUT / folder; target.mkdir(exist_ok=False)
    receipts = []
    for row in rows:
        body = fetch('https://raw.githubusercontent.com/' + REPO + '/' +
                     manifest['tree_sha'] + '/' + row['path'],
                     folder + '/' + Path(row['path']).name)
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if len(body) != row['size'] or blob != row['sha']:
            raise ValueError('Publisher configuration mismatch')
        receipts.append({'path': row['path'], 'bytes': len(body),
                         'sha256': hashlib.sha256(body).hexdigest()})
    (target / 'receipts.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps(receipts))


if __name__ == '__main__':
    if '--session' in sys.argv[1:]:
        configuration(session=True)
    elif '--context' in sys.argv[1:]:
        configuration(context=True)
    elif '--cleanup' in sys.argv[1:]:
        configuration(cleanup=True)
    elif '--configuration' in sys.argv[1:]:
        configuration()
    elif '--sources' in sys.argv[1:]:
        sources()
    else:
        main()
