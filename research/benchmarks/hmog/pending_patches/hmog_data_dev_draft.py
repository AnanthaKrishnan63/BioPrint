"""Fail-closed HMOG role guard. No archive extraction or implicit cohort access."""
from contextlib import contextmanager
import hashlib
import json
import pathlib
import re
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / 'datasets/hmog'
MANIFEST_SHA256 = 'f1e23826b5ad7d750f26406134d22f3dd9571055d0ac9834b8606ac90f912a90'
SCHEMA_V2_SHA256 = '749bceb59afa61fc507ebb267110bb122883569dfa5e86b630b86f5f1e0f321c'
FEATURE_FEASIBILITY_SHA256 = '0ee0375d0d9a6fbc9fc2dacca82b52183041e824e99b090bb0d70c366fc29678'
FEATURE_EXTRACTION_SHA256 = 'd1120f6e38fd358ddb80026947b731dfb6d3ea9f6891ccb5af1b5a0df80a7eda'
SELECTION_EXTRACTION_SHA256 = '3a55ac0a851e35af7e072cc0dba6335d480825918ee807742713ba1d41cb0343'
SELECTION_V2_EXTRACTION_SHA256 = '40675f80c9a022e763d96e1fec3db8c37201282789ed9ba5af76543b280a8313'
CALIBRATION_V2_SHA256 = '3571933f4a4af4f5cb44ffe4003ab26d09e22149f3ae7fb60e6537fc79d3fa7c'
DEV_V2_SHA256 = None  # Activation awaits completed calibration provenance.
DEV_SUBJECTS = frozenset(('556357','219303','777078','737973'))
CALIBRATION_SUBJECTS = frozenset(('962159', '663153'))
SELECTION_SUBJECTS = frozenset(('180679', '622852'))
ALL_FIT_SUBJECTS = frozenset(('717868', '526319', '986737', '539502'))
# Deliberately narrow current authorization; broadening requires a recorded instruction.
EXPLORATION_SUBJECT = '717868'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def create_manifest():
    """Receipt metadata only; exclusive creation before any measurement access."""
    prereg = BASE / 'preregistered_acquisition.json'
    p = json.loads(prereg.read_text())
    entries = []
    hashes = {'preregistered_acquisition.json': digest(prereg)}
    archives = {}
    for subject in p['cohort']:
        sid = pathlib.Path(subject['name']).stem
        cohort = subject['identity_role']
        relative = f'receipts/{sid}.zip.json'
        receipt = BASE / relative
        hashes[relative] = digest(receipt)
        r = json.loads(receipt.read_text())
        assert r['subject'] == sid and r['identity_role'] == cohort
        archives[sid] = {'sha256': r['sha256'], 'bytes': r['raw_inner_zip_bytes'], 'cohort': cohort}
        for m in r['inner_member_metadata']:
            match = re.fullmatch(rf'{sid}/{sid}_session_([1-9][0-9]*)/([^/]+\.csv)', m['name'])
            if not match:
                continue
            session = int(match.group(1))
            if session >= 17:
                role = 'sealed_test'
            elif session <= 8:
                role = 'train_support' if cohort == 'dev' else 'train_enrollment'
            else:
                role = {'fit': 'train_fit', 'selection': 'train_selection', 'calibration': 'train_calibration', 'dev': 'dev_probe'}[cohort]
            entries.append({'subject': sid, 'cohort': cohort, 'session': session, 'role': role, **m})
    assert len({e['name'] for e in entries}) == len(entries)
    document = {'version': 1, 'scope': 'Receipt-derived exact CSV whitelist; all sessions >=17 sealed test. Unknown paths denied.',
                'session_order_caveat': 'Numeric session order does not establish chronological days.',
                'source_hashes': hashes, 'archives': archives, 'members': entries,
                'unfetched_test_ids': [pathlib.Path(m['name']).stem for m in p['sealed_test_never_fetch']],
                'unused_sealed_ids': [pathlib.Path(m['name']).stem for m in p['unused_sealed_never_fetch']],
                'current_authorization': {'subject': EXPLORATION_SUBJECT, 'cohort': 'fit', 'sessions': '1-16', 'purpose': 'schema_inspection'},
                'population_fit_rule': 'Only fit identity train_fit records may fit global parameters. Enrollment/support cannot fit population parameters.'}
    path = BASE / 'session_roles.json'
    with path.open('x') as f:
        json.dump(document, f, indent=2, sort_keys=True)
    return digest(path)


class HmogReader:
    def __init__(self, *, phase='first_fit_schema'):
        if phase not in ('first_fit_schema', 'all_fit_schema', 'fit_feature_feasibility', 'fit_feature_extraction', 'selection_feature_extraction', 'selection_v2_feature_extraction', 'calibration_v2_feature_extraction', 'dev_v2_feature_extraction'):
            raise PermissionError('Unknown named authorization phase')
        if phase == 'calibration_v2_feature_extraction' and CALIBRATION_V2_SHA256 is None:
            raise PermissionError('Calibration authorization has not been issued')
        if phase == 'dev_v2_feature_extraction' and DEV_V2_SHA256 is None:
            raise PermissionError('Dev authorization has not been issued')
        self.phase = phase
        manifest = BASE / 'session_roles.json'
        manifest_bytes = manifest.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError('HMOG session manifest checksum mismatch')
        self.manifest = json.loads(manifest_bytes)
        for relative, expected in self.manifest['source_hashes'].items():
            if digest(BASE / relative) != expected:
                raise ValueError('HMOG acquisition/receipt checksum mismatch')
        self.entries = {m['name']: m for m in self.manifest['members']}
        self.allowed_subjects = frozenset((EXPLORATION_SUBJECT,))
        self.allowed_purpose = 'schema_inspection'
        self.allowed_cohort = 'fit'
        self.allowed_roles = frozenset(('train_enrollment', 'train_fit'))
        if phase in ('all_fit_schema', 'fit_feature_feasibility', 'fit_feature_extraction'):
            overlay_name, expected_hash = ('schema_authorization_v2.json', SCHEMA_V2_SHA256)
            if phase == 'fit_feature_feasibility':
                overlay_name, expected_hash = ('feature_feasibility_authorization_v1.json', FEATURE_FEASIBILITY_SHA256)
                self.allowed_purpose = 'feature_feasibility'
            if phase == 'fit_feature_extraction':
                overlay_name, expected_hash = ('feature_extraction_authorization_v1.json', FEATURE_EXTRACTION_SHA256)
                self.allowed_purpose = 'feature_extraction'
            overlay_bytes = (BASE / overlay_name).read_bytes()
            if hashlib.sha256(overlay_bytes).hexdigest() != expected_hash:
                raise ValueError('HMOG schema authorization checksum mismatch')
            overlay = json.loads(overlay_bytes)
            for relative, expected in overlay['source_hashes'].items():
                if digest(BASE / relative) != expected:
                    raise ValueError('HMOG schema authorization source checksum mismatch')
            if (overlay['phase'] != phase or set(overlay['allowed_subjects']) != ALL_FIT_SUBJECTS
                    or overlay['allowed_cohort'] != 'fit' or overlay['allowed_sessions'] != list(range(1, 17))
                    or set(overlay['allowed_roles']) != {'train_enrollment', 'train_fit'}
                    or overlay['allowed_purpose'] != self.allowed_purpose):
                raise ValueError('HMOG schema authorization exceeds fixed phase bounds')
            self.allowed_subjects = ALL_FIT_SUBJECTS
        if phase in ('selection_feature_extraction', 'selection_v2_feature_extraction'):
            version = 2 if phase == 'selection_v2_feature_extraction' else 1
            expected_hash = SELECTION_V2_EXTRACTION_SHA256 if version == 2 else SELECTION_EXTRACTION_SHA256
            expected_purpose = 'selection_feature_extraction_v2' if version == 2 else 'selection_feature_extraction'
            overlay_bytes = (BASE / f'selection_feature_extraction_authorization_v{version}.json').read_bytes()
            if hashlib.sha256(overlay_bytes).hexdigest() != expected_hash:
                raise ValueError('HMOG selection authorization checksum mismatch')
            overlay = json.loads(overlay_bytes)
            for relative, expected in overlay['source_hashes'].items():
                if digest(BASE / relative) != expected:
                    raise ValueError('HMOG selection authorization source checksum mismatch')
            protocol = overlay['model_protocol']
            if (protocol['path'] != f'research/benchmarks/hmog/model_protocol_v{version}.json'
                    or digest(ROOT / protocol['path']) != protocol['sha256']):
                raise ValueError('HMOG selection model protocol checksum mismatch')
            if (overlay['phase'] != phase or set(overlay['allowed_subjects']) != SELECTION_SUBJECTS
                    or overlay['allowed_cohort'] != 'selection' or overlay['allowed_sessions'] != list(range(1,17))
                    or set(overlay['allowed_roles']) != {'train_enrollment','train_selection'}
                    or overlay['allowed_purpose'] != expected_purpose):
                raise ValueError('HMOG selection authorization exceeds fixed phase bounds')
            self.allowed_subjects = SELECTION_SUBJECTS
            self.allowed_cohort = 'selection'
            self.allowed_roles = frozenset(('train_enrollment','train_selection'))
            self.allowed_purpose = expected_purpose
        if phase == 'calibration_v2_feature_extraction':
            overlay_bytes = (BASE / 'calibration_feature_extraction_authorization_v2.json').read_bytes()
            if hashlib.sha256(overlay_bytes).hexdigest() != CALIBRATION_V2_SHA256:
                raise ValueError('HMOG calibration authorization checksum mismatch')
            overlay = json.loads(overlay_bytes)
            for relative, expected in overlay['source_hashes'].items():
                if digest(BASE / relative) != expected:
                    raise ValueError('HMOG calibration authorization source checksum mismatch')
            protocol = overlay['model_protocol']
            if (protocol['path'] != 'research/benchmarks/hmog/model_protocol_v2.json'
                    or digest(ROOT / protocol['path']) != protocol['sha256']):
                raise ValueError('HMOG calibration model protocol checksum mismatch')
            selection = overlay['selection']
            report_path = (ROOT / selection['report_path']).resolve()
            checkpoint_path = (ROOT / selection['checkpoint_path']).resolve()
            if (not report_path.is_relative_to(ROOT.resolve()) or not checkpoint_path.is_relative_to(ROOT.resolve())
                    or report_path.name != 'selection_complete.json' or checkpoint_path.name != 'selected_checkpoint.pt'
                    or report_path.parent != checkpoint_path.parent):
                raise ValueError('Invalid pinned selection artifact paths')
            raw = report_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != selection['report_sha256']:
                raise ValueError('HMOG completed selection checksum mismatch')
            report = json.loads(raw)
            if (selection['status'] != 'complete' or report.get('status') != 'complete'
                    or report.get('selection_only') is not True or report.get('calibration_performed') is not False
                    or report.get('dev_accessed') is not False
                    or report.get('plan', {}).get('model_protocol_sha256') != protocol['sha256']
                    or report.get('selected_checkpoint_sha256') != selection['checkpoint_sha256']
                    or digest(checkpoint_path) != selection['checkpoint_sha256']):
                raise ValueError('HMOG selection is not complete with the pinned checkpoint')
            if (overlay['phase'] != phase or set(overlay['allowed_subjects']) != CALIBRATION_SUBJECTS
                    or overlay['allowed_cohort'] != 'calibration' or overlay['allowed_sessions'] != list(range(1,17))
                    or set(overlay['allowed_roles']) != {'train_enrollment','train_calibration'}
                    or overlay['allowed_purpose'] != 'calibration_feature_extraction_v2'):
                raise ValueError('HMOG calibration authorization exceeds fixed phase bounds')
            self.allowed_subjects = CALIBRATION_SUBJECTS
            self.allowed_cohort = 'calibration'
            self.allowed_roles = frozenset(('train_enrollment','train_calibration'))
            self.allowed_purpose = 'calibration_feature_extraction_v2'

        if phase == 'dev_v2_feature_extraction':
            raw = (BASE / 'dev_feature_extraction_authorization_v2.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != DEV_V2_SHA256:
                raise ValueError('HMOG dev authorization checksum mismatch')
            overlay = json.loads(raw)
            for relative, expected in overlay['source_hashes'].items():
                if digest(BASE / relative) != expected:
                    raise ValueError('HMOG dev authorization source checksum mismatch')
            protocol = overlay['model_protocol']
            if (protocol['path'] != 'research/benchmarks/hmog/model_protocol_v2.json'
                    or digest(ROOT / protocol['path']) != protocol['sha256']):
                raise ValueError('HMOG dev model protocol checksum mismatch')
            calibration = overlay['calibration']
            report_path = (ROOT / calibration['report_path']).resolve()
            thresholds_path = (ROOT / calibration['thresholds_path']).resolve()
            checkpoint_path = (ROOT / overlay['selection_checkpoint']['path']).resolve()
            if (any(not p.is_relative_to(ROOT.resolve()) for p in (report_path, thresholds_path, checkpoint_path))
                    or report_path.name != 'calibration_complete.json' or thresholds_path.name != 'thresholds.json'
                    or report_path.parent != thresholds_path.parent or checkpoint_path.name != 'selected_checkpoint.pt'):
                raise ValueError('Invalid pinned calibration artifact paths')
            raw = report_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != calibration['report_sha256']:
                raise ValueError('HMOG completed calibration checksum mismatch')
            report = json.loads(raw)
            checkpoint_sha = overlay['selection_checkpoint']['sha256']
            if (calibration['status'] != 'complete' or report.get('status') != 'complete'
                    or report.get('global_model_fitted') is not False or report.get('dev_accessed') is not False
                    or report.get('plan', {}).get('model_protocol_sha256') != protocol['sha256']
                    or report.get('selected_checkpoint_sha256') != checkpoint_sha
                    or report.get('thresholds_sha256') != calibration['thresholds_sha256']
                    or digest(thresholds_path) != calibration['thresholds_sha256']
                    or digest(checkpoint_path) != checkpoint_sha):
                raise ValueError('HMOG calibration is not complete with pinned thresholds/checkpoint')
            if (overlay['phase'] != phase or set(overlay['allowed_subjects']) != DEV_SUBJECTS
                    or overlay['allowed_cohort'] != 'dev' or overlay['allowed_sessions'] != list(range(1,17))
                    or set(overlay['allowed_roles']) != {'train_support','dev_probe'}
                    or overlay['allowed_purpose'] != 'dev_feature_extraction_v2'):
                raise ValueError('HMOG dev authorization exceeds fixed phase bounds')
            self.allowed_subjects = DEV_SUBJECTS
            self.allowed_cohort = 'dev'
            self.allowed_roles = frozenset(('train_support','dev_probe'))
            self.allowed_purpose = 'dev_feature_extraction_v2'

    def authorize(self, subject, session, filename, *, expected_cohort, expected_role, purpose):
        subject = str(subject)
        if type(session) is not int or session < 1 or session >= 17:
            raise PermissionError('Unknown or sealed TEST session')
        if not isinstance(filename, str) or '/' in filename or '\\' in filename or filename in ('.', '..'):
            raise PermissionError('Invalid exact member filename')
        path = f'{subject}/{subject}_session_{session}/{filename}'
        entry = self.entries.get(path)
        if entry is None or entry['subject'] != subject:
            raise PermissionError('Unknown or held-out identity/member')
        if entry['cohort'] != expected_cohort or entry['role'] != expected_role:
            raise PermissionError('Explicit cohort/record-role mismatch')
        if purpose == 'population_fit' and (entry['cohort'] != 'fit' or entry['role'] != 'train_fit'):
            raise PermissionError('Enrollment/support/non-fit observations cannot fit population parameters')
        if subject not in self.allowed_subjects or entry['cohort'] != self.allowed_cohort or purpose != self.allowed_purpose:
            raise PermissionError('Current named phase permits only its designated TRAIN cohort/purpose')
        if entry['role'] not in self.allowed_roles:
            raise PermissionError('Current authorization excludes this record role')
        return entry

    @contextmanager
    def open_member(self, subject, session, filename, *, expected_cohort, expected_role, purpose):
        """Yield a binary member only after exact role, provenance, archive and index checks."""
        entry = self.authorize(subject, session, filename, expected_cohort=expected_cohort, expected_role=expected_role, purpose=purpose)
        sid = entry['subject']; metadata = self.manifest['archives'][sid]
        # Hash and open the same file handle; never deserialize an unchecked replacement.
        with (BASE / 'raw' / f'{sid}.zip').open('rb') as f:
            h = hashlib.sha256(); count = 0
            for chunk in iter(lambda: f.read(1048576), b''):
                count += len(chunk); h.update(chunk)
            if count != metadata['bytes'] or h.hexdigest() != metadata['sha256']:
                raise ValueError('HMOG raw archive checksum/size mismatch')
            f.seek(0)
            with zipfile.ZipFile(f) as archive:
                infos = [i for i in archive.infolist() if i.filename == entry['name']]
                if len(infos) != 1:
                    raise ValueError('Missing/duplicate exact member')
                info = infos[0]
                if (info.file_size, info.compress_size, info.CRC) != (entry['uncompressed_bytes'], entry['compressed_bytes'], entry['crc32']):
                    raise ValueError('HMOG member metadata mismatch')
                with archive.open(info, 'r') as member:
                    yield member

    def iter_lines(self, subject, session, filename, *, expected_cohort, expected_role, purpose, max_lines=10, max_line_bytes=65536):
        """Bounded UTF-8 text sample; no contents cached or extracted to disk."""
        if type(max_lines) is not int or not 1 <= max_lines <= 1000 or type(max_line_bytes) is not int or not 1 <= max_line_bytes <= 1048576:
            raise ValueError('Invalid bounded sample limits')
        with self.open_member(subject, session, filename, expected_cohort=expected_cohort, expected_role=expected_role, purpose=purpose) as member:
            for _ in range(max_lines):
                line = member.readline(max_line_bytes + 1)
                if not line:
                    break
                if len(line) > max_line_bytes:
                    raise ValueError('Line exceeds explicit sample bound')
                yield line.decode('utf-8-sig')


if __name__ == '__main__':
    print(create_manifest())
