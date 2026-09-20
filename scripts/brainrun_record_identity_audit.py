"""Audit game/gesture identity metadata without decoding behavioral values."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from brainrun_bson_metadata import iter_metadata, MAX_DOCUMENT_BYTES
from brainrun_guarded_stream_v2 import canonical, _validate_identifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets/brainrun'
OUT = ROOT/'research/benchmarks/brainrun_record_identity_audit_v1'
ARCHIVE_SHA256 = '070f43c7a068e96e484d303ad67e85d018859aebbf1c25d6e5b7fe563426bb6e'
ROLES_SHA256 = '22c69bd7562f47d6ddbc73f4641387f3f9e3367773e0fcf99ab03e6a66d82183'


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            checksum.update(block)
    return checksum.hexdigest()


class IdentityAudit:
    def __init__(self, device_to_user, quarantine_device_ids, known_users):
        self.mapping = dict(device_to_user)
        self.quarantine = frozenset(quarantine_device_ids)
        self.users = frozenset(known_users)
        for value in [*self.mapping, *self.mapping.values(), *self.quarantine, *self.users]:
            _validate_identifier(value)
        if self.quarantine & self.mapping.keys() or not set(self.mapping.values()) <= self.users:
            raise ValueError('Frozen mapping/quarantine contract mismatch')
        self.counts = Counter({key: 0 for key in ['records','known','quarantine','unknown','missing_device',
                                                'conflicting_user','explicit_user_present','explicit_user_missing',
                                                'explicit_user_outside_frozen_users']})
        self.unknown = defaultdict(lambda: {'records': 0, 'explicit_users': set(), 'missing_user_records': 0})
        self.conflicts = defaultdict(Counter)

    def observe(self, metadata):
        if not set(metadata) <= {'device_id','user_id'}:
            raise ValueError('Identity-only metadata expected')
        self.counts['records'] += 1
        user = canonical(metadata['user_id']) if 'user_id' in metadata else None
        if user is None:
            self.counts['explicit_user_missing'] += 1
        else:
            _validate_identifier(user)
            self.counts['explicit_user_present'] += 1
            self.counts['explicit_user_outside_frozen_users'] += user not in self.users
        if 'device_id' not in metadata:
            self.counts['missing_device'] += 1
            return
        device = canonical(metadata['device_id']); _validate_identifier(device)
        if device in self.quarantine:
            self.counts['quarantine'] += 1
        elif device in self.mapping:
            self.counts['known'] += 1
            if user is not None and user != self.mapping[device]:
                self.counts['conflicting_user'] += 1
                self.conflicts[device][user] += 1
        else:
            self.counts['unknown'] += 1
            entry = self.unknown[device]; entry['records'] += 1
            if user is None:entry['missing_user_records'] += 1
            else:entry['explicit_users'].add(user)

    def report(self):
        if self.counts['records'] != sum(self.counts[k] for k in ['known','quarantine','unknown','missing_device']):
            raise ValueError('Record accounting mismatch')
        return {'counts': dict(self.counts), 'unknown_device_count': len(self.unknown),
                'unknown_devices': {device: {**entry, 'explicit_users': sorted(entry['explicit_users'])}
                                    for device,entry in sorted(self.unknown.items())},
                'conflicting_device_users': {device: dict(sorted(users.items()))
                                             for device,users in sorted(self.conflicts.items())}}


def main():
    roles_path = DATA/'identity_roles_v2/roles.json'
    if sha(roles_path) != ROLES_SHA256:
        raise ValueError('Frozen v2 identity roles changed')
    roles = json.loads(roles_path.read_text())
    role_plan_path = roles_path.parent/'plan.json'
    role_plan = json.loads(role_plan_path.read_text())
    if (roles['status'] != 'identity_metadata_roles_frozen'
            or roles['behavioral_observations_decoded'] is not False
            or role_plan['archive_sha256'] != ARCHIVE_SHA256):
        raise ValueError('Frozen metadata-only identity assignment required')
    for name,expected in role_plan['source_sha256'].items():
        if sha(ROOT/name) != expected:raise ValueError('Role source changed')
    receipt_path = DATA/'acquisition_v1/report.json'
    receipt = json.loads(receipt_path.read_text()); archive = ROOT/receipt['archive']
    if (receipt['status'] != 'compressed_archive_acquired' or receipt['observations_decoded'] is not False
            or receipt['sha256'] != ARCHIVE_SHA256 or sha(archive) != ARCHIVE_SHA256):
        raise ValueError('Verified original compressed archive required')
    sources = [Path(__file__),roles_path,role_plan_path,receipt_path,
               ROOT/'scripts/brainrun_bson_metadata.py',ROOT/'scripts/brainrun_guarded_stream_v2.py']
    plan = {'scope': 'All games/gestures identity metadata only; no behavioral value decoding or cohort selection',
            'fields': ['device_id','user_id'], 'collections': ['games','gestures'],
            'max_document_bytes': MAX_DOCUMENT_BYTES, 'archive_sha256': ARCHIVE_SHA256,
            'roles_sha256': ROLES_SHA256, 'full_extraction': False,
            'unknown_mapping_policy': 'Report explicit typed identity associations only; do not repair mappings or authorize observations',
            'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sources}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    collections = {}
    with zipfile.ZipFile(archive) as zipped:
        for collection in plan['collections']:
            audit = IdentityAudit(roles['device_to_user'],roles['quarantine_device_ids'],roles['users'])
            member = 'gestures_devices_users_games_data/'+collection+'.bson'
            with zipped.open(member) as stream:
                for metadata in iter_metadata(stream,fields=plan['fields']):
                    audit.observe(metadata)
            # Iteration consumes each member to EOF, invoking ZipExtFile CRC validation.
            collections[collection] = {**audit.report(), 'member': member,
                                       'member_crc_verified_at_eof': True}
            print(json.dumps({'collection':collection, **audit.report()['counts']}),flush=True)
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name) != expected:raise ValueError('Audit source changed during run')
    if sha(archive) != ARCHIVE_SHA256:raise ValueError('Archive changed during audit')
    result = {'status': 'record_identifier_metadata_audit_complete', 'collections': collections,
              'archive_sha256': ARCHIVE_SHA256, 'roles_sha256': ROLES_SHA256,
              'behavioral_observations_decoded': False, 'context_decoded': False,
              'identity_metadata_includes_sealed_roles': True, 'recognition_metrics': {}}
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status']}),flush=True)


if __name__ == '__main__':main()
