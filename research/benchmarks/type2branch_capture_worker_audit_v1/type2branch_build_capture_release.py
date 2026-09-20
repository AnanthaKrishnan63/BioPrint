"""Package completed capture DEV evaluation for isolated local API replay."""
import json
from pathlib import Path
import shutil
import numpy as np
from type2branch_continuity import ROOT,sha
from type2branch_calibrate_lengths import verify_hashes
from type2branch_capture_release import RELEASE,FILES,SOURCES,verify_release
from type2branch_capture_wire import validate_request


def main():
    dev=ROOT/'research/benchmarks/type2branch_capture_dev_evaluation_v1'
    report=json.loads((dev/'report.json').read_text())
    if report['status']!='frozen_capture_dev_evaluation_complete':
        raise ValueError('Completed capture DEV evaluation required')
    plan=json.loads((dev/'plan.json').read_text());verify_hashes(plan)
    verify_hashes(plan['protocol'])
    scorefile=dev/'scores.npz'
    if sha(scorefile)!=report['scores_sha256']:raise ValueError('DEV scores changed')
    cal=ROOT/'research/benchmarks/type2branch_length_calibration_v1'
    calibration_plan=json.loads((cal/'plan.json').read_text());verify_hashes(calibration_plan)
    if calibration_plan['checkpoint']!=plan['checkpoint']:raise ValueError('Checkpoint binding mismatch')
    checkpoint=Path(plan['checkpoint']['checkpoint'])
    for name,expected in calibration_plan['checkpoint_sha256'].items():
        if sha(checkpoint.parent/name)!=expected:raise ValueError('Checkpoint changed')
    verify_hashes(json.loads((ROOT/'research/benchmarks/type2branch_train_v1/plan.json').read_text()))
    thresholds={str(k):float(v) for k,v in plan['thresholds'].items()}
    if set(thresholds)!={'25','50','75','100'} or not np.isfinite(list(thresholds.values())).all():
        raise ValueError('Four finite calibrated thresholds required')
    with np.load(scorefile,allow_pickle=False) as data:
        subjects=data['subjects'];gallery=data['gallery_embeddings'];probe_subjects=data['probe_subjects']
        probe_lengths=data['probe_lengths'];scores=data['scores'];decisions=data['decisions']
        if (gallery.shape!=(79,5,256) or subjects.shape!=(79,) or len(set(subjects))!=79
                or not np.isfinite(gallery).all() or scores.shape!=(len(probe_subjects),79)):
            raise ValueError('Complete profile/score contract mismatch')
        expected_thresholds=np.array([thresholds[str(int(n))] for n in probe_lengths])
        np.testing.assert_array_equal(data['thresholds'],expected_thresholds)
        np.testing.assert_array_equal(decisions,scores>=expected_thresholds[:,None])
    features=ROOT/'research/benchmarks/type2branch_capture_dev_features_v1/probes.npz'
    if sha(features)!=plan['input_sha256'][str(features.relative_to(ROOT))]:raise ValueError('DEV features changed')
    with np.load(features,allow_pickle=False) as data:
        original=data['features'];lengths=data['true_length'];ids=data['subject'];windows=data['window']
        if (original.shape!=(len(ids),100,5) or ids.ndim!=1 or lengths.shape!=ids.shape
                or windows.shape!=ids.shape or windows.dtype.kind not in 'iu' or not (windows==0).all()):
            raise ValueError('Exact first-capture metadata shapes required')
        batches=[]
        for start in range(0,len(ids),32):
            _,batch,_=validate_request({'id':start,'features':original[start:start+32].tolist(),
                                      'true_lengths':lengths[start:start+32].tolist()})
            batches.append(batch)
        x=np.concatenate(batches) if batches else np.empty((0,100,5),dtype=np.float32)
        np.testing.assert_array_equal(ids,probe_subjects)
        np.testing.assert_array_equal(lengths,probe_lengths)
    RELEASE.mkdir(exist_ok=False)
    anchors=np.array([25,50,75,100],dtype=np.int64)
    np.savez_compressed(RELEASE/'profiles.npz',gallery=gallery,subjects=subjects,lengths=anchors,
                        thresholds=np.array([thresholds[str(n)] for n in anchors],dtype=np.float64))
    np.savez_compressed(RELEASE/'dev_features.npz',features=x,subject=ids,window=windows,true_length=lengths)
    shutil.copyfile(str(checkpoint)+'.index',RELEASE/'encoder.index')
    shutil.copyfile(str(checkpoint)+'.data-00000-of-00001',RELEASE/'encoder.data-00000-of-00001')
    shutil.copyfile(dev/'report.json',RELEASE/'dev_report.json')
    manifest={'version':1,'protocol':'capture','split':'dev','subjects':subjects.tolist(),'thresholds':thresholds,
        'checkpoint_updates':plan['checkpoint']['updates'],'checkpoint_epoch':plan['checkpoint']['epoch'],
        'eligible_probes':len(ids),'files':{name:sha(RELEASE/name) for name in sorted(FILES)},
        'sources':{name:sha(ROOT/name) for name in sorted(SOURCES)},
        'provenance':{'dev_report_sha256':sha(dev/'report.json'),'dev_scores_sha256':sha(scorefile),
                      'calibration_plan_sha256':sha(cal/'plan.json')},
        'limitations':report['limitations'],'prior_exposure':report['prior_exposure']}
    (RELEASE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pin=sha(RELEASE/'manifest.json');verify_release(pin)
    print(json.dumps({'status':'capture_release_built_not_activated','manifest_sha256':pin}))


if __name__=='__main__':main()
