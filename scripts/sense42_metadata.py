"""Fetch bounded publisher metadata only; never follow measurement-file links."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'zenodo': 'https://zenodo.org/api/records/20328099',
    'figshare': 'https://api.figshare.com/v2/articles/29971873',
    'readme': 'https://raw.githubusercontent.com/Catherine9811/HCI-SENSE-42/main/README.md',
}
SYNAPSE_SOURCES = {
    'synapse_entity': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68713182',
    'synapse_wiki': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68713182/wiki/633562',
    'synapse_children': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/children',
    'readme': 'https://raw.githubusercontent.com/Catherine9811/HCI-SENSE-42/master/README.md',
}


def main():
    label = sys.argv[1]
    if not label.isalnum():
        raise ValueError('Use an alphanumeric audit label')
    out = ROOT / 'datasets/joint_feasibility_metadata' / ('sense42_' + label)
    out.mkdir(exist_ok=False)
    receipts = []
    sources = SYNAPSE_SOURCES if '--synapse' in sys.argv[2:] else SOURCES
    if '--behavioral' in sys.argv[2:]:
        sources = {'synapse_children': SYNAPSE_SOURCES['synapse_children'],
                   'requirements': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68714674/accessRequirement'}
    if '--formats' in sys.argv[2:]:
        sources = {name: SYNAPSE_SOURCES['synapse_children'] for name in
                   ('synapse_children_csv', 'synapse_children_psydat')}
    if '--handle' in sys.argv[2:]:
        sources = {
            'csv_handle': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68723176/filehandles',
            'csv_requirements': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68723176/accessRequirement',
        }
    if '--permissions' in sys.argv[2:]:
        sources = {
            'csv_permissions': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68723176/permissions',
            'csv_entity': 'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn68723176',
        }
    for name, url in sources.items():
        row = dict(name=name, url=url, checked_at=datetime.datetime.now(
            datetime.timezone.utc).isoformat(), scope='metadata/documentation only')
        try:
            request = url
            if name.startswith('synapse_children'):
                parent = {'synapse_children_csv': 'syn68714788',
                          'synapse_children_psydat': 'syn68714789'}.get(name,
                          'syn68714674' if '--behavioral' in sys.argv[2:] else 'syn68713182')
                request = urllib.request.Request(url, data=json.dumps({
                    'parentId': parent,
                    'includeTypes': ['folder', 'file'],
                    'includeTotalChildCount': True,
                }).encode(), headers={'Content-Type': 'application/json'})
                row['method'] = 'POST (read-only metadata listing)'
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    raise ValueError('Metadata exceeds 2 MB bound')
                row.update(status=response.status, bytes=len(body),
                           sha256=hashlib.sha256(body).hexdigest())
            (out / (name + '.txt')).write_bytes(body)
            if name != 'readme':
                data = json.loads(body)
                row['access'] = data.get('access', data.get('metadata', {}).get('access_right'))
                row['files_metadata'] = data.get('files')
        except Exception as exc:
            row['error'] = str(exc)
        receipts.append(row)
        (out / 'receipt.json').write_text(json.dumps(receipts, indent=2))
        summary = {k: v for k, v in row.items() if k != 'files_metadata'}
        if isinstance(row.get('files_metadata'), list):
            summary['file_count'] = len(row['files_metadata'])
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
