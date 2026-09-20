"""Package only completed frozen DEV artifacts for local API replay."""
import json
from pathlib import Path
import shutil
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_release import RELEASE, FILES, SOURCES, verify_release


def main():
    dev=ROOT/'research/benchmarks/type2branch_dev_v1'
    report=json.loads((dev/'report.json').read_text())
    if report['status']!='frozen_cross_session_dev_complete':raise ValueError('Completed DEV evaluation required')
    plan=json.loads((dev/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('DEV source/input provenance changed')
    scorefile=dev/'dev_scores.npz'
    if sha(scorefile)!=report['output_sha256']:raise ValueError('DEV scores changed')
    cal=ROOT/'research/benchmarks/type2branch_calibration_v1'
    calibrated=json.loads((cal/'plan.json').read_text())
    if calibrated['checkpoint']!=plan['checkpoint']:raise ValueError('Checkpoint mismatch')
    checkpoint=Path(plan['checkpoint']['checkpoint'])
    for name,expected in calibrated['checkpoint_sha256'].items():
        if sha(checkpoint.parent/name)!=expected:raise ValueError('Checkpoint changed')
    features=ROOT/'research/benchmarks/type2branch_dev_features_v1/probes.npz'
    if sha(features)!=plan['feature_sha256']['probes']:raise ValueError('DEV probe features changed')
    with np.load(scorefile,allow_pickle=False) as a:
        subjects=a['subjects']; gallery=[]
        for subject in subjects:
            rows=np.flatnonzero(a['gallery_subjects']==subject)
            rows=rows[np.argsort(a['gallery_windows'][rows])]
            if not np.array_equal(a['gallery_windows'][rows],np.arange(15,20)):
                raise ValueError('Gallery window allocation changed')
            gallery.append(a['gallery_embeddings'][rows])
        gallery=np.stack(gallery);threshold=a['threshold']
    if gallery.shape!=(79,5,256) or not np.isfinite(gallery).all():raise ValueError('Invalid gallery')
    if threshold.ndim!=0 or float(threshold)!=plan['threshold']:raise ValueError('Threshold mismatch')
    # Exclusivity preserves prior releases; this tool does not activate an API pin.
    RELEASE.mkdir(exist_ok=False)
    np.savez_compressed(RELEASE/'profiles.npz',gallery=gallery,subjects=subjects,threshold=threshold)
    shutil.copyfile(str(checkpoint)+'.index',RELEASE/'encoder.index')
    shutil.copyfile(str(checkpoint)+'.data-00000-of-00001',RELEASE/'encoder.data-00000-of-00001')
    with np.load(features,allow_pickle=False) as a:
        # Offline inference uses this same float32 cast; wire release is explicit.
        np.savez_compressed(RELEASE/'dev_features.npz',features=a['features'].astype(np.float32),
                            subject=a['subject'],window=a['window'],true_length=a['true_length'])
    shutil.copyfile(dev/'report.json',RELEASE/'dev_report.json')
    manifest={'version':1,'split':'dev','subjects':subjects.tolist(),'threshold':float(threshold),
              'checkpoint_updates':plan['checkpoint']['updates'],'checkpoint_epoch':plan['checkpoint']['epoch'],
              'files':{name:sha(RELEASE/name) for name in sorted(FILES)},
              'sources':{name:sha(ROOT/name) for name in sorted(SOURCES)},
              'provenance':{'dev_report_sha256':sha(dev/'report.json'),'dev_scores_sha256':sha(scorefile),
                            'calibration_plan_sha256':sha(cal/'plan.json')},
              'limitations':report['limitations']}
    (RELEASE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    digest=sha(RELEASE/'manifest.json')
    verify_release(digest)
    print(json.dumps({'status':'release_built_not_activated','manifest_sha256':digest}))


if __name__=='__main__':main()
