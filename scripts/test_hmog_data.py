import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('hmog_data', pathlib.Path(__file__).with_name('hmog_data.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class RoleGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name); (self.base / 'raw').mkdir()
        archive = self.base / 'raw/717868.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('717868/717868_session_1/TouchEvent.csv', 'synthetic header\nsynthetic row\n')
            z.writestr('717868/717868_session_17/TouchEvent.csv', 'POISON TEST NEVER READ')
        entries = []
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                session = 17 if '_17/' in info.filename else 1
                entries.append({'name': info.filename, 'subject': '717868', 'cohort': 'fit', 'session': session, 'role': 'sealed_test' if session==17 else 'train_enrollment', 'compressed_bytes': info.compress_size, 'uncompressed_bytes': info.file_size, 'crc32': info.CRC})
        for sid, cohort, role in [('219303','dev','train_support'),('180679','selection','train_enrollment')]:
            entries.append(dict(entries[0],name=f'{sid}/{sid}_session_1/TouchEvent.csv',subject=sid,cohort=cohort,role=role))
        (self.base / 'receipt.json').write_text('{}')
        doc = {'source_hashes': {'receipt.json': m.digest(self.base / 'receipt.json')}, 'archives': {'717868': {'sha256': m.digest(archive), 'bytes': archive.stat().st_size}}, 'members': entries}
        other = self.base / 'raw/526319.zip'
        with zipfile.ZipFile(other, 'w') as z:
            z.writestr('526319/526319_session_9/TouchEvent.csv', 'synthetic second fit\n')
        with zipfile.ZipFile(other) as z:
            info = z.infolist()[0]
            entries.append({'name':info.filename,'subject':'526319','cohort':'fit','session':9,'role':'train_fit','compressed_bytes':info.compress_size,'uncompressed_bytes':info.file_size,'crc32':info.CRC})
        doc['archives']['526319'] = {'sha256':m.digest(other),'bytes':other.stat().st_size}
        selection=self.base/'raw/180679.zip'
        with zipfile.ZipFile(selection,'w') as z:z.writestr('180679/180679_session_1/TouchEvent.csv','synthetic selection\n')
        with zipfile.ZipFile(selection) as z:
            info=z.infolist()[0]
            for entry in entries:
                if entry['subject']=='180679':entry.update(compressed_bytes=info.compress_size,uncompressed_bytes=info.file_size,crc32=info.CRC)
        doc['archives']['180679']={'sha256':m.digest(selection),'bytes':selection.stat().st_size}
        (self.base / 'session_roles.json').write_text(json.dumps(doc))
        overlay = {'phase':'all_fit_schema','source_hashes':{'session_roles.json':m.digest(self.base/'session_roles.json')},'allowed_subjects':sorted(m.ALL_FIT_SUBJECTS),'allowed_cohort':'fit','allowed_sessions':list(range(1,17)),'allowed_roles':['train_enrollment','train_fit'],'allowed_purpose':'schema_inspection'}
        (self.base/'schema_authorization_v2.json').write_text(json.dumps(overlay))
        feasibility = dict(overlay,phase='fit_feature_feasibility',allowed_purpose='feature_feasibility')
        (self.base/'feature_feasibility_authorization_v1.json').write_text(json.dumps(feasibility))
        extraction = dict(overlay,phase='fit_feature_extraction',allowed_purpose='feature_extraction')
        (self.base/'feature_extraction_authorization_v1.json').write_text(json.dumps(extraction))
        protocol=self.base/'research/benchmarks/hmog/model_protocol_v1.json'
        protocol.parent.mkdir(parents=True);protocol.write_text('{"synthetic":true}')
        selected=dict(overlay,phase='selection_feature_extraction',allowed_subjects=['180679','622852'],allowed_cohort='selection',allowed_roles=['train_enrollment','train_selection'],allowed_purpose='selection_feature_extraction',model_protocol={'path':'research/benchmarks/hmog/model_protocol_v1.json','sha256':m.digest(protocol)})
        (self.base/'selection_feature_extraction_authorization_v1.json').write_text(json.dumps(selected))
        protocol2=protocol.with_name('model_protocol_v2.json');protocol2.write_text('{"synthetic_v2":true}')
        selected2=dict(selected,phase='selection_v2_feature_extraction',allowed_purpose='selection_feature_extraction_v2',model_protocol={'path':'research/benchmarks/hmog/model_protocol_v2.json','sha256':m.digest(protocol2)})
        (self.base/'selection_feature_extraction_authorization_v2.json').write_text(json.dumps(selected2))
        self.bp = patch.object(m, 'BASE', self.base); self.bp.start()
        self.hp = patch.object(m, 'MANIFEST_SHA256', m.digest(self.base / 'session_roles.json')); self.hp.start()
        self.op = patch.object(m, 'SCHEMA_V2_SHA256', m.digest(self.base/'schema_authorization_v2.json')); self.op.start()
        self.fp = patch.object(m, 'FEATURE_FEASIBILITY_SHA256', m.digest(self.base/'feature_feasibility_authorization_v1.json')); self.fp.start()
        self.ep = patch.object(m, 'FEATURE_EXTRACTION_SHA256', m.digest(self.base/'feature_extraction_authorization_v1.json')); self.ep.start()
        self.sp=patch.object(m,'SELECTION_EXTRACTION_SHA256',m.digest(self.base/'selection_feature_extraction_authorization_v1.json'));self.sp.start()
        self.rp=patch.object(m,'ROOT',self.base);self.rp.start()
        self.s2p=patch.object(m,'SELECTION_V2_EXTRACTION_SHA256',m.digest(self.base/'selection_feature_extraction_authorization_v2.json'));self.s2p.start()
    def tearDown(self):
        self.s2p.stop();self.rp.stop();self.sp.stop();self.ep.stop(); self.fp.stop(); self.op.stop(); self.hp.stop(); self.bp.stop(); self.tmp.cleanup()
    def kwargs(self): return {'expected_cohort':'fit','expected_role':'train_enrollment','purpose':'schema_inspection'}
    def test_allowed_synthetic_sample(self):
        self.assertEqual(list(m.HmogReader().iter_lines('717868',1,'TouchEvent.csv',**self.kwargs(),max_lines=1)),['synthetic header\n'])
    def test_forbidden_before_any_zip_member_open(self):
        cases=[('717868',17,'TouchEvent.csv',self.kwargs()),('000000',1,'TouchEvent.csv',self.kwargs()),('717868',1,'unknown.csv',self.kwargs()),('717868',1,'../TouchEvent.csv',self.kwargs()),('717868',1,'TouchEvent.csv',dict(self.kwargs(),expected_role='train_fit')),('717868',1,'TouchEvent.csv',dict(self.kwargs(),expected_cohort='dev')),('219303',1,'TouchEvent.csv',dict(expected_cohort='dev',expected_role='train_support',purpose='population_fit')),('180679',1,'TouchEvent.csv',dict(expected_cohort='selection',expected_role='train_enrollment',purpose='schema_inspection'))]
        reader=m.HmogReader()
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            for sid,session,name,k in cases:
                with self.subTest(sid=sid,session=session,name=name,kwargs=k):
                    with self.assertRaises(PermissionError): list(reader.iter_lines(sid,session,name,**k))
            opened.assert_not_called()
    def test_manifest_tamper(self):
        with (self.base/'session_roles.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'manifest checksum'):m.HmogReader()
    def test_receipt_tamper(self):
        (self.base/'receipt.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'receipt checksum'):m.HmogReader()
    def test_archive_tamper_before_member_read(self):
        with (self.base/'raw/717868.zip').open('ab') as f:f.write(b'bad')
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            with self.assertRaisesRegex(ValueError,'archive checksum'):list(m.HmogReader().iter_lines('717868',1,'TouchEvent.csv',**self.kwargs()))
            opened.assert_not_called()
    def test_generated_manifest_roles_metadata_only(self):
        # Real immutable role metadata is safe; do not open any real archive.
        real = pathlib.Path(__file__).resolve().parents[1]/'datasets/hmog/session_roles.json'
        doc=json.loads(real.read_text())
        for row in doc['members']:
            if row['session']>=17:self.assertEqual(row['role'],'sealed_test')
            if row['cohort']=='dev' and row['session']<=8:self.assertEqual(row['role'],'train_support')
        self.assertEqual(len(doc['archives']),12)
        self.assertEqual(len(doc['unfetched_test_ids']),4)
        self.assertEqual(len(doc['unused_sealed_ids']),84)
    def test_second_fit_requires_explicit_v2(self):
        kwargs=dict(self.kwargs(),expected_role='train_fit')
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            with self.assertRaises(PermissionError):list(m.HmogReader().iter_lines('526319',9,'TouchEvent.csv',**kwargs))
            opened.assert_not_called()
        self.assertEqual(list(m.HmogReader(phase='all_fit_schema').iter_lines('526319',9,'TouchEvent.csv',**kwargs)),['synthetic second fit\n'])
    def test_v2_does_not_authorize_test_other_cohorts_or_fitting(self):
        r=m.HmogReader(phase='all_fit_schema')
        cases=[('526319',17,'fit','sealed_test','schema_inspection'),('219303',1,'dev','train_support','schema_inspection'),('180679',1,'selection','train_enrollment','schema_inspection'),('526319',9,'fit','train_fit','population_fit'),('000000',1,'fit','train_enrollment','schema_inspection')]
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            for sid,session,cohort,role,purpose in cases:
                with self.subTest(sid=sid,session=session,purpose=purpose):
                    with self.assertRaises(PermissionError):list(r.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
            opened.assert_not_called()
    def test_overlay_tamper(self):
        with (self.base/'schema_authorization_v2.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'authorization checksum'):m.HmogReader(phase='all_fit_schema')
        m.HmogReader()  # Narrow default does not rely on the overlay.
    def test_unknown_phase_rejected(self):
        with self.assertRaises(PermissionError):m.HmogReader(phase='dev')
    def test_feasibility_explicit_phase_and_purpose(self):
        kwargs=dict(self.kwargs(),expected_role='train_fit',purpose='feature_feasibility')
        r=m.HmogReader(phase='fit_feature_feasibility')
        self.assertEqual(list(r.iter_lines('526319',9,'TouchEvent.csv',**kwargs)),['synthetic second fit\n'])
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            for phase in ['first_fit_schema','all_fit_schema']:
                with self.assertRaises(PermissionError):list(m.HmogReader(phase=phase).iter_lines('526319',9,'TouchEvent.csv',**kwargs))
            for sid,session,cohort,role,purpose in [('717868',17,'fit','sealed_test','feature_feasibility'),('219303',1,'dev','train_support','feature_feasibility'),('180679',1,'selection','train_enrollment','feature_feasibility'),('526319',9,'fit','train_fit','schema_inspection'),('526319',9,'fit','train_fit','population_fit'),('000000',1,'fit','train_enrollment','feature_feasibility')]:
                with self.subTest(sid=sid,purpose=purpose):
                    with self.assertRaises(PermissionError):list(r.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
            opened.assert_not_called()
    def test_feasibility_overlay_tamper(self):
        with (self.base/'feature_feasibility_authorization_v1.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'authorization checksum'):m.HmogReader(phase='fit_feature_feasibility')
    def test_extraction_phase_separate_from_fitting_and_other_phases(self):
        kwargs=dict(self.kwargs(),expected_role='train_fit',purpose='feature_extraction')
        reader=m.HmogReader(phase='fit_feature_extraction')
        self.assertEqual(list(reader.iter_lines('526319',9,'TouchEvent.csv',**kwargs)),['synthetic second fit\n'])
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            for phase in ['first_fit_schema','all_fit_schema','fit_feature_feasibility']:
                with self.assertRaises(PermissionError):list(m.HmogReader(phase=phase).iter_lines('526319',9,'TouchEvent.csv',**kwargs))
            for sid,session,cohort,role,purpose in [('717868',17,'fit','sealed_test','feature_extraction'),('219303',1,'dev','train_support','feature_extraction'),('180679',1,'selection','train_enrollment','feature_extraction'),('526319',9,'fit','train_fit','population_fit'),('000000',1,'fit','train_enrollment','feature_extraction')]:
                with self.subTest(sid=sid,purpose=purpose):
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
            opened.assert_not_called()
    def test_extraction_overlay_tamper(self):
        with (self.base/'feature_extraction_authorization_v1.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'authorization checksum'):m.HmogReader(phase='fit_feature_extraction')
    def test_selection_only_explicit_phase(self):
        kwargs=dict(expected_cohort='selection',expected_role='train_enrollment',purpose='selection_feature_extraction')
        reader=m.HmogReader(phase='selection_feature_extraction')
        self.assertEqual(list(reader.iter_lines('180679',1,'TouchEvent.csv',**kwargs)),['synthetic selection\n'])
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            with self.assertRaises(PermissionError):list(m.HmogReader().iter_lines('180679',1,'TouchEvent.csv',**kwargs))
            for sid,session,cohort,role,purpose in [('717868',1,'fit','train_enrollment','selection_feature_extraction'),('219303',1,'dev','train_support','selection_feature_extraction'),('180679',17,'selection','sealed_test','selection_feature_extraction'),('180679',1,'selection','train_enrollment','population_fit'),('180679',1,'selection','train_enrollment','feature_extraction')]:
                with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
            opened.assert_not_called()
    def test_selection_protocol_tamper(self):
        (self.base/'research/benchmarks/hmog/model_protocol_v1.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'protocol checksum'):m.HmogReader(phase='selection_feature_extraction')
    def test_selection_overlay_tamper(self):
        with (self.base/'selection_feature_extraction_authorization_v1.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'authorization checksum'):m.HmogReader(phase='selection_feature_extraction')
    def test_selection_v2_phase_and_cohort_boundaries(self):
        kwargs=dict(expected_cohort='selection',expected_role='train_enrollment',purpose='selection_feature_extraction_v2')
        reader=m.HmogReader(phase='selection_v2_feature_extraction')
        self.assertEqual(list(reader.iter_lines('180679',1,'TouchEvent.csv',**kwargs)),['synthetic selection\n'])
        with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
            for phase in ['first_fit_schema','selection_feature_extraction']:
                with self.assertRaises(PermissionError):list(m.HmogReader(phase=phase).iter_lines('180679',1,'TouchEvent.csv',**kwargs))
            for sid,session,cohort,role,purpose in [('717868',1,'fit','train_enrollment','selection_feature_extraction_v2'),('219303',1,'dev','train_support','selection_feature_extraction_v2'),('180679',17,'selection','sealed_test','selection_feature_extraction_v2'),('180679',1,'selection','train_enrollment','population_fit')]:
                with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
            opened.assert_not_called()
    def test_selection_v2_protocol_tamper(self):
        (self.base/'research/benchmarks/hmog/model_protocol_v2.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'protocol checksum'):m.HmogReader(phase='selection_v2_feature_extraction')
        m.HmogReader(phase='selection_feature_extraction')
    def calibration_fixture(self, *, status='complete'):
        manifest_path=self.base/'session_roles.json';doc=json.loads(manifest_path.read_text())
        template=doc['members'][0]
        doc['members'].append(dict(template,name='663153/663153_session_9/TouchEvent.csv',subject='663153',cohort='calibration',session=9,role='train_calibration'))
        manifest_path.write_text(json.dumps(doc));self.hp.stop()
        self.hp=patch.object(m,'MANIFEST_SHA256',m.digest(manifest_path));self.hp.start()
        destination=self.base/'synthetic_selection';destination.mkdir()
        checkpoint=destination/'selected_checkpoint.pt';checkpoint.write_bytes(b'synthetic-checkpoint-no-deserialization')
        protocol=self.base/'research/benchmarks/hmog/model_protocol_v2.json'
        report={'status':status,'selection_only':True,'calibration_performed':False,'dev_accessed':False,'plan':{'model_protocol_sha256':m.digest(protocol)},'selected_checkpoint_sha256':m.digest(checkpoint)}
        rp=destination/'selection_complete.json';rp.write_text(json.dumps(report))
        overlay={'phase':'calibration_v2_feature_extraction','source_hashes':{'session_roles.json':m.digest(manifest_path)},'model_protocol':{'path':'research/benchmarks/hmog/model_protocol_v2.json','sha256':m.digest(protocol)},'selection':{'status':'complete','report_path':'synthetic_selection/selection_complete.json','report_sha256':m.digest(rp),'checkpoint_path':'synthetic_selection/selected_checkpoint.pt','checkpoint_sha256':m.digest(checkpoint)},'allowed_subjects':['962159','663153'],'allowed_cohort':'calibration','allowed_sessions':list(range(1,17)),'allowed_roles':['train_enrollment','train_calibration'],'allowed_purpose':'calibration_feature_extraction_v2'}
        path=self.base/'calibration_feature_extraction_authorization_v2.json';path.write_text(json.dumps(overlay))
        return path,checkpoint
    def test_calibration_unissued_denied_before_metadata_read(self):
        with patch.object(m,'CALIBRATION_V2_SHA256',None),patch.object(pathlib.Path,'read_bytes',side_effect=AssertionError('NO READ')):
            with self.assertRaisesRegex(PermissionError,'not been issued'):m.HmogReader(phase='calibration_v2_feature_extraction')
    def test_calibration_phase_bounds_with_synthetic_issued_overlay(self):
        path,_=self.calibration_fixture()
        with patch.object(m,'CALIBRATION_V2_SHA256',m.digest(path)):
            reader=m.HmogReader(phase='calibration_v2_feature_extraction')
            self.assertEqual(reader.authorize('663153',9,'TouchEvent.csv',expected_cohort='calibration',expected_role='train_calibration',purpose='calibration_feature_extraction_v2')['subject'],'663153')
            with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
                for sid,session,cohort,role,purpose in [('663153',17,'calibration','sealed_test','calibration_feature_extraction_v2'),('219303',1,'dev','train_support','calibration_feature_extraction_v2'),('180679',1,'selection','train_enrollment','calibration_feature_extraction_v2'),('717868',1,'fit','train_enrollment','calibration_feature_extraction_v2'),('663153',9,'calibration','train_calibration','population_fit')]:
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
                opened.assert_not_called()
    def test_calibration_rejects_incomplete_selection(self):
        path,_=self.calibration_fixture(status='infeasible')
        with patch.object(m,'CALIBRATION_V2_SHA256',m.digest(path)):
            with self.assertRaisesRegex(ValueError,'not complete'):m.HmogReader(phase='calibration_v2_feature_extraction')
    def test_calibration_rejects_checkpoint_tamper(self):
        path,checkpoint=self.calibration_fixture();checkpoint.write_bytes(b'changed')
        with patch.object(m,'CALIBRATION_V2_SHA256',m.digest(path)):
            with self.assertRaisesRegex(ValueError,'pinned checkpoint'):m.HmogReader(phase='calibration_v2_feature_extraction')
    def dev_fixture(self, *, status='complete'):
        protocol=self.base/'research/benchmarks/hmog/model_protocol_v2.json'
        destination=self.base/'synthetic_calibration';destination.mkdir()
        thresholds=destination/'thresholds.json';thresholds.write_text('{"joint":{"0.01":1.0}}')
        selection=self.base/'synthetic_selection';selection.mkdir()
        checkpoint=selection/'selected_checkpoint.pt';checkpoint.write_bytes(b'synthetic-checkpoint')
        report={'status':status,'global_model_fitted':False,'dev_accessed':False,'plan':{'model_protocol_sha256':m.digest(protocol)},'selected_checkpoint_sha256':m.digest(checkpoint),'thresholds_sha256':m.digest(thresholds)}
        policy=protocol.with_name('calibration_fallback_protocol_v1.json');policy.write_text('{"synthetic_fallback":true}')
        report['plan']['calibration_protocol_sha256']=m.digest(policy)
        report['calibration_protocol_sha256']=m.digest(policy)
        rp=destination/'calibration_complete.json';rp.write_text(json.dumps(report))
        overlay={'phase':'dev_v2_feature_extraction','source_hashes':{'session_roles.json':m.digest(self.base/'session_roles.json')},'model_protocol':{'path':'research/benchmarks/hmog/model_protocol_v2.json','sha256':m.digest(protocol)},'calibration':{'status':'complete','report_path':'synthetic_calibration/calibration_complete.json','report_sha256':m.digest(rp),'thresholds_path':'synthetic_calibration/thresholds.json','thresholds_sha256':m.digest(thresholds)},'selection_checkpoint':{'path':'synthetic_selection/selected_checkpoint.pt','sha256':m.digest(checkpoint)},'allowed_subjects':['556357','219303','777078','737973'],'allowed_cohort':'dev','allowed_sessions':list(range(1,17)),'allowed_roles':['train_support','dev_probe'],'allowed_purpose':'dev_feature_extraction_v2'}
        overlay['calibration_protocol']={'path':'research/benchmarks/hmog/calibration_fallback_protocol_v1.json','sha256':m.digest(policy)}
        p=self.base/'dev_feature_extraction_authorization_v2.json';p.write_text(json.dumps(overlay))
        return p,thresholds
    def test_dev_unissued_denied_before_metadata_read(self):
        with patch.object(m,'DEV_V2_SHA256',None),patch.object(pathlib.Path,'read_bytes',side_effect=AssertionError('NO READ')):
            with self.assertRaisesRegex(PermissionError,'not been issued'):m.HmogReader(phase='dev_v2_feature_extraction')
    def test_dev_support_role_and_no_population_fit(self):
        p,_=self.dev_fixture()
        with patch.object(m,'DEV_V2_SHA256',m.digest(p)):
            reader=m.HmogReader(phase='dev_v2_feature_extraction')
            self.assertEqual(reader.authorize('219303',1,'TouchEvent.csv',expected_cohort='dev',expected_role='train_support',purpose='dev_feature_extraction_v2')['role'],'train_support')
            with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
                for sid,session,cohort,role,purpose in [('219303',17,'dev','sealed_test','dev_feature_extraction_v2'),('180679',1,'selection','train_enrollment','dev_feature_extraction_v2'),('717868',1,'fit','train_enrollment','dev_feature_extraction_v2'),('219303',1,'dev','train_support','population_fit'),('219303',1,'dev','train_enrollment','dev_feature_extraction_v2')]:
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,'TouchEvent.csv',expected_cohort=cohort,expected_role=role,purpose=purpose))
                opened.assert_not_called()
    def test_dev_threshold_tamper_rejected(self):
        p,t=self.dev_fixture();t.write_text('{}')
        with patch.object(m,'DEV_V2_SHA256',m.digest(p)):
            with self.assertRaisesRegex(ValueError,'pinned thresholds'):m.HmogReader(phase='dev_v2_feature_extraction')
    def test_dev_incomplete_calibration_rejected(self):
        p,_=self.dev_fixture(status='infeasible')
        with patch.object(m,'DEV_V2_SHA256',m.digest(p)):
            with self.assertRaisesRegex(ValueError,'not complete'):m.HmogReader(phase='dev_v2_feature_extraction')
    def test_dev_fallback_policy_tamper_rejected(self):
        p,_=self.dev_fixture()
        (self.base/'research/benchmarks/hmog/calibration_fallback_protocol_v1.json').write_text('{}')
        with patch.object(m,'DEV_V2_SHA256',m.digest(p)):
            with self.assertRaisesRegex(ValueError,'policy checksum'):m.HmogReader(phase='dev_v2_feature_extraction')
    def tap_fixture(self):
        manifest=self.base/'session_roles.json';doc=json.loads(manifest.read_text())
        for name in ['Activity.csv','TouchEvent_im.csv']:
            doc['members'].append(dict(doc['members'][0],name='717868/717868_session_1/'+name))
        manifest.write_text(json.dumps(doc));self.hp.stop();self.hp=patch.object(m,'MANIFEST_SHA256',m.digest(manifest));self.hp.start()
        protocol=self.base/'research/benchmarks/hmog_tap_protocol_proposal_v1.json';protocol.write_text('{}')
        archive=self.base/'datasets/hmog/fit_features_v3.npz';archive.parent.mkdir(parents=True);archive.write_bytes(b'syntheticopaque')
        overlay={'phase':'tap_reference_feature_extraction','allowed_purpose':'tap_reference_feature_extraction','allowed_subjects':sorted(m.ALL_FIT_SUBJECTS),'allowed_cohort':'fit','allowed_sessions':list(range(1,17)),'allowed_roles':['train_enrollment','train_fit'],'allowed_filenames':['Activity.csv','TouchEvent_im.csv'],'source_hashes':{'session_roles.json':m.digest(manifest)},'tap_protocol':{'path':'research/benchmarks/hmog_tap_protocol_proposal_v1.json','sha256':m.digest(protocol)},'window_archive':{'path':'datasets/hmog/fit_features_v3.npz','sha256':m.digest(archive)}}
        path=self.base/'tap_reference_feature_extraction_authorization_v1.json';path.write_text(json.dumps(overlay))
        return path,archive
    def test_tap_phase_exact_members_cohort_and_purpose(self):
        path,_=self.tap_fixture();kwargs=dict(expected_cohort='fit',expected_role='train_enrollment',purpose='tap_reference_feature_extraction')
        with patch.object(m,'TAP_REFERENCE_SHA256',m.digest(path)):
            reader=m.HmogReader(phase='tap_reference_feature_extraction')
            for name in ['Activity.csv','TouchEvent_im.csv']:reader.authorize('717868',1,name,**kwargs)
            with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
                for sid,session,name,k in [('717868',1,'KeyPressEvent.csv',kwargs),('717868',1,'TouchEvent.csv',kwargs),('717868',17,'TouchEvent_im.csv',kwargs),('219303',1,'TouchEvent_im.csv',dict(kwargs,expected_cohort='dev',expected_role='train_support')),('717868',1,'TouchEvent_im.csv',dict(kwargs,purpose='feature_extraction'))]:
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,name,**k))
                opened.assert_not_called()
    def test_tap_window_archive_tamper(self):
        path,archive=self.tap_fixture();archive.write_bytes(b'changed')
        with patch.object(m,'TAP_REFERENCE_SHA256',m.digest(path)):
            with self.assertRaisesRegex(ValueError,'protocol/window checksum'):m.HmogReader(phase='tap_reference_feature_extraction')
    def selection_tap_fixture(self):
        manifest=self.base/'session_roles.json';doc=json.loads(manifest.read_text())
        doc['members'].append(dict(doc['members'][0],name='180679/180679_session_1/TouchEvent_im.csv',subject='180679',cohort='selection',role='train_enrollment'))
        manifest.write_text(json.dumps(doc));self.hp.stop();self.hp=patch.object(m,'MANIFEST_SHA256',m.digest(manifest));self.hp.start()
        paths={'selection_tap_protocol':'research/benchmarks/hmog/tap_selection_protocol_v1.json','window_archive':'datasets/hmog/selection_features_v1.npz','paired_report':'research/benchmarks/hmog/tap_paired_train_v2/paired_train_complete.json'}
        for rel in paths.values():
            p=self.base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic')
        o={'phase':'tap_selection_feature_extraction','allowed_purpose':'tap_selection_feature_extraction','allowed_subjects':sorted(m.SELECTION_SUBJECTS),'allowed_cohort':'selection','allowed_sessions':list(range(1,17)),'allowed_roles':['train_enrollment','train_selection'],'allowed_filenames':['Activity.csv','TouchEvent_im.csv'],'source_hashes':{'session_roles.json':m.digest(manifest)}}
        o.update({k:{'path':v,'sha256':m.digest(self.base/v)} for k,v in paths.items()})
        p=self.base/'tap_selection_feature_extraction_authorization_v1.json';p.write_text(json.dumps(o));return p,paths
    def test_selection_tap_members_and_roles(self):
        p,_=self.selection_tap_fixture()
        with patch.object(m,'TAP_SELECTION_SHA256',m.digest(p)):
            reader=m.HmogReader(phase='tap_selection_feature_extraction')
            reader.authorize('180679',1,'TouchEvent_im.csv',expected_cohort='selection',expected_role='train_enrollment',purpose='tap_selection_feature_extraction')
            with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
                for sid,session,name,cohort,role,purpose in [('180679',1,'KeyPressEvent.csv','selection','train_enrollment','tap_selection_feature_extraction'),('180679',17,'TouchEvent_im.csv','selection','sealed_test','tap_selection_feature_extraction'),('717868',1,'TouchEvent_im.csv','fit','train_enrollment','tap_selection_feature_extraction'),('219303',1,'TouchEvent_im.csv','dev','train_support','tap_selection_feature_extraction'),('180679',1,'TouchEvent_im.csv','selection','train_enrollment','population_fit')]:
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,name,expected_cohort=cohort,expected_role=role,purpose=purpose))
                opened.assert_not_called()
    def test_selection_tap_parameter_tamper(self):
        p,paths=self.selection_tap_fixture();(self.base/paths['paired_report']).write_bytes(b'changed')
        with patch.object(m,'TAP_SELECTION_SHA256',m.digest(p)):
            with self.assertRaisesRegex(ValueError,'paired-report checksum'):m.HmogReader(phase='tap_selection_feature_extraction')
    def dev_tap_fixture(self):
        manifest=self.base/'session_roles.json';doc=json.loads(manifest.read_text())
        doc['members'].append(dict(doc['members'][0],name='219303/219303_session_1/TouchEvent_im.csv',subject='219303',cohort='dev',role='train_support'))
        manifest.write_text(json.dumps(doc));self.hp.stop();self.hp=patch.object(m,'MANIFEST_SHA256',m.digest(manifest));self.hp.start()
        protocol=self.base/'research/benchmarks/hmog/tap_dev_evaluation_protocol_v1.json';protocol.write_text('{}')
        params=self.base/'synthetic_fit_parameters.json';params.write_text('{}')
        o={'phase':'tap_dev_evaluation_feature_extraction','allowed_purpose':'tap_dev_evaluation_feature_extraction','allowed_subjects':sorted(m.DEV_SUBJECTS),'allowed_cohort':'dev','allowed_sessions':list(range(1,17)),'allowed_roles':['train_support','dev_probe'],'allowed_filenames':['Activity.csv','TouchEvent_im.csv'],'source_hashes':{'session_roles.json':m.digest(manifest)},'evaluation_protocol':{'path':'research/benchmarks/hmog/tap_dev_evaluation_protocol_v1.json','sha256':m.digest(protocol)},'bindings':{'fit_parameters':{'path':'synthetic_fit_parameters.json','sha256':m.digest(params)}},'helper_sources':{}}
        p=self.base/'tap_dev_evaluation_feature_extraction_authorization_v1.json';p.write_text(json.dumps(o));return p,params
    def test_dev_tap_exact_scope(self):
        p,_=self.dev_tap_fixture()
        with patch.object(m,'TAP_DEV_EVALUATION_SHA256',m.digest(p)):
            reader=m.HmogReader(phase='tap_dev_evaluation_feature_extraction')
            reader.authorize('219303',1,'TouchEvent_im.csv',expected_cohort='dev',expected_role='train_support',purpose='tap_dev_evaluation_feature_extraction')
            with patch.object(zipfile.ZipFile,'open',side_effect=AssertionError('POISON READ')) as opened:
                for sid,session,name,cohort,role,purpose in [('219303',1,'KeyPressEvent.csv','dev','train_support','tap_dev_evaluation_feature_extraction'),('219303',17,'TouchEvent_im.csv','dev','sealed_test','tap_dev_evaluation_feature_extraction'),('180679',1,'TouchEvent_im.csv','selection','train_enrollment','tap_dev_evaluation_feature_extraction'),('717868',1,'TouchEvent_im.csv','fit','train_enrollment','tap_dev_evaluation_feature_extraction'),('219303',1,'TouchEvent_im.csv','dev','train_support','population_fit')]:
                    with self.assertRaises(PermissionError):list(reader.iter_lines(sid,session,name,expected_cohort=cohort,expected_role=role,purpose=purpose))
                opened.assert_not_called()
    def test_dev_tap_fit_parameter_tamper(self):
        p,params=self.dev_tap_fixture();params.write_text('{"changed":true}')
        with patch.object(m,'TAP_DEV_EVALUATION_SHA256',m.digest(p)):
            with self.assertRaisesRegex(ValueError,'parameters/window checksum'):m.HmogReader(phase='tap_dev_evaluation_feature_extraction')

if __name__=='__main__':unittest.main()
