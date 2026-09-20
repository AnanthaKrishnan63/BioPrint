"""Future calibration-only CLI. Import and tests never read participant data."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np

from hmog_select import BRANCHES, digest_bytes, read_json, load_checkpoint, embed_all
from hmog_training_core import load_author_classes, MODEL_HASH, LOSS_HASH
from hmog_verification import mean_gallery, euclidean_scores, pooled_eer, far_threshold, verification_rates

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/benchmarks/hmog/model_protocol_v2.json'
CALIBRATION_IDS=frozenset(('962159','663153'))
TARGETS=(.001,.01,.05)


def load_calibration_arrays(path):
    """Metadata authorization precedes any feature array decompression/copy."""
    with np.load(path,allow_pickle=False) as archive:
        subjects,sessions,roles=(archive[k].copy() for k in ('subject','session','role'))
        if (subjects.ndim!=1 or subjects.dtype.kind not in 'US' or roles.shape!=subjects.shape
                or roles.dtype.kind not in 'US' or sessions.shape!=subjects.shape or sessions.dtype.kind not in 'iu'):
            raise ValueError('Invalid calibration metadata')
        subjects,roles=subjects.astype(str),roles.astype(str)
        if not set(subjects).issubset(CALIBRATION_IDS):raise PermissionError('Non-calibration identity')
        for session,role in zip(sessions,roles):
            expected='train_enrollment' if 1<=session<=8 else 'train_calibration'
            if not 1<=session<=16 or role!=expected:raise PermissionError('Unauthorized calibration session/role')
        key,imu=archive['key'].copy(),archive['imu'].copy()
    n=len(subjects)
    if (key.shape!=(n,50,10) or imu.shape!=(n,100,24) or key.dtype!=np.float32 or imu.dtype!=np.float32
            or not np.isfinite(key).all() or not np.isfinite(imu).all()):raise ValueError('Invalid calibration feature arrays')
    return key,imu,subjects,sessions,roles


def missing_accounts(subjects,roles):
    return [{'subject':s,'missing_role':r} for s in sorted(CALIBRATION_IDS)
            for r in ('train_enrollment','train_calibration') if not np.any((subjects==s)&(roles==r))]


def coverage_from_report(report,subjects,sessions):
    inspected=report.get('subjects',{});excluded=report.get('excluded_sessions',{})
    if set(inspected)!=CALIBRATION_IDS or not set(excluded).issubset(CALIBRATION_IDS):
        raise ValueError('Coverage must identify exactly both calibration accounts')
    result={}
    for sid in sorted(CALIBRATION_IDS):
        observed=inspected[sid];unknown=excluded.get(sid,{})
        if set(observed)&set(unknown) or set(observed)|set(unknown)!={str(i) for i in range(1,17)}:
            raise ValueError('Each session1–16 must be inspected or explicitly metadata-excluded')
        result[sid]={}
        for role,ss in [('train_enrollment',range(1,9)),('train_calibration',range(9,17))]:
            candidate=eligible=0;uninspected=[]
            for session in ss:
                number=int(np.sum((subjects==sid)&(sessions==session)))
                if str(session) in unknown:
                    row=unknown[str(session)]
                    if (number or row.get('basis')!='archive_metadata' or row.get('activity_contents_inspected') is not False
                            or row.get('reason') not in ('missing_keypress','empty_keypress')
                            or row.get('candidate_windows') is not None or row.get('eligible_windows') is not None):
                        raise ValueError('Invalid metadata exclusion')
                    uninspected.append({'session':session,'reason':row['reason']});continue
                row=observed[str(session)];c,e=row.get('candidate_windows'),row.get('eligible_windows')
                if type(c)is not int or type(e)is not int or not 0<=e<=c or e!=number:
                    raise ValueError('Coverage/feature row count mismatch')
                candidate+=c;eligible+=e
            result[sid][role]={'known_candidate_windows':candidate,'eligible_windows':eligible,
                'known_excluded_windows':candidate-eligible,'uninspected_sessions':uninspected,
                'total_candidate_windows':None if uninspected else candidate,
                'conditional_known_window_coverage':eligible/candidate if candidate else None}
    return result


def validate_selection(report,protocol_sha,expected_report_sha,actual_report_sha):
    if actual_report_sha!=expected_report_sha:raise ValueError('Selection report checksum mismatch')
    if (report.get('status')!='complete' or report.get('selection_only')is not True
            or report.get('calibration_performed')is not False or report.get('dev_accessed')is not False
            or report.get('plan',{}).get('model_protocol_sha256')!=protocol_sha):
        raise ValueError('Selection must be complete and frozen before calibration')
    chosen=report.get('selected_epoch');history=report.get('history',[])
    if type(chosen)is not int or len(history)!=20 or [r.get('epoch') for r in history]!=list(range(1,21)):
        raise ValueError('Require frozen twenty-epoch selection history')
    values=[r['metrics']['joint']['pooled_eer'] for r in history]
    if not all(np.isfinite(v) and 0<=v<=1 for v in values):raise ValueError('Invalid selection EER')
    best=min(history,key=lambda r:(r['metrics']['joint']['pooled_eer'],r['epoch']))
    expected=report.get('selected_checkpoint_sha256')
    if (chosen!=best['epoch'] or best.get('checkpoint_sha256')!=expected or not isinstance(expected,str)
            or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('Selected checkpoint disagrees with frozen selection objective')
    return expected


def calibrate_embeddings(embeddings,subjects,roles):
    if not set(subjects).issubset(CALIBRATION_IDS) or set(roles)-{'train_enrollment','train_calibration'}:
        raise PermissionError('Calibration-only embeddings required')
    if missing_accounts(subjects,roles):raise ValueError('Both calibration accounts need enrollment and probes')
    enroll=roles=='train_enrollment';probe=roles=='train_calibration'
    accounts=np.array(sorted(CALIBRATION_IDS));mask=subjects[probe,None]==accounts[None,:]
    arrays={'accounts':accounts,'probe_subject':subjects[probe],'genuine_mask':mask};metrics={}
    for branch in BRANCHES:
        vectors=np.asarray(embeddings[branch])
        if vectors.shape!=(len(subjects),64) or not np.isfinite(vectors).all():raise ValueError('Invalid branch embeddings')
        gallery=mean_gallery(vectors[enroll],subjects[enroll]);names,scores=euclidean_scores(vectors[probe],gallery)
        if names!=accounts.tolist():raise ValueError('Unexpected gallery account order')
        g,i=scores[mask],scores[~mask]
        ops={}
        for target in TARGETS:
            threshold=far_threshold(i,target)
            ops[str(target)]={'threshold':threshold,**verification_rates(g,i,threshold),
                'per_account':{sid:verification_rates(scores[subjects[probe]==sid,j],scores[subjects[probe]!=sid,j],threshold)
                               for j,sid in enumerate(names)}}
        metrics[branch]={'pooled_eer_diagnostic':pooled_eer(g,i),'genuine_count':len(g),'impostor_count':len(i),
                         'empirical_far_resolution':1/len(i),'operating_points':ops}
        arrays[branch+'_distances']=scores
        arrays[branch+'_gallery']=np.asarray([gallery[s] for s in names])
    return metrics,arrays


def main():
    p=argparse.ArgumentParser()
    for name in ('selection-dir','features','extraction-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection-report-sha256',required=True)
    args=p.parse_args()
    for path in (args.selection_dir,args.features,args.extraction_report,args.output):
        if not path.resolve().is_relative_to(ROOT):raise ValueError('All paths must remain in repository')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol,protocol_sha=read_json(PROTOCOL)
    if protocol.get('version')!=2:raise ValueError('Expected model protocol v2')
    selection,selection_sha=read_json(args.selection_dir/'selection_complete.json')
    checkpoint_sha=validate_selection(selection,protocol_sha,args.selection_report_sha256,selection_sha)
    extraction,extraction_sha=read_json(args.extraction_report)
    feature_sha=digest_bytes(args.features.read_bytes())
    if (extraction.get('feature_archive_sha256')!=feature_sha
            or extraction.get('plan',{}).get('model_protocol_sha256')!=protocol_sha):
        raise ValueError('Extraction report must bind exact features and model protocol')
    sources={name:(ROOT/'scripts'/name).read_bytes() for name in
             ('hmog_calibrate.py','hmog_select.py','hmog_verification.py','hmog_training_core.py')}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'model_protocol_sha256':protocol_sha,
          'selection_report_sha256':selection_sha,'selected_checkpoint_sha256':checkpoint_sha,
          'extraction_report_sha256':extraction_sha,'features_sha256':feature_sha,
          'sources_sha256':{n:digest_bytes(v) for n,v in sources.items()},'author_model_sha256':MODEL_HASH,
          'author_loss_sha256':LOSS_HASH,'calibration_ids':sorted(CALIBRATION_IDS),'targets':TARGETS,
          'decision':'distance <= threshold','batch_size':8,'cpu_threads':2,'dev_or_test_accessed':False,
          'branch_caveat':'Three branches from the same joint-selected encoder; no branch-specific selection'}
    (args.output/'calibration_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name,raw in sources.items():(args.output/('source_'+name)).write_bytes(raw)
    # Durable plan precedes all feature-value decompression, copying, or inference.
    key,imu,subjects,sessions,roles=load_calibration_arrays(args.features)
    if digest_bytes(args.features.read_bytes())!=feature_sha:raise ValueError('Features changed during loading')
    coverage=coverage_from_report(extraction,subjects,sessions)
    missing=missing_accounts(subjects,roles)
    if missing:
        (args.output/'calibration_complete.json').write_text(json.dumps({'status':'infeasible','missing':missing,'coverage':coverage,'plan':plan},indent=2)+'\n');return
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260920)
    model_class,_=load_author_classes();model=model_class(10,24,50,100,64).eval()
    load_checkpoint(model,args.selection_dir/'selected_checkpoint.pt',checkpoint_sha)
    metrics,arrays=calibrate_embeddings(embed_all(model,key,imu),subjects,roles)
    if digest_bytes(args.features.read_bytes())!=feature_sha:raise ValueError('Features changed during calibration')
    with (args.output/'calibration_scores.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    thresholds={branch:{str(target):metrics[branch]['operating_points'][str(target)]['threshold']
                        for target in TARGETS} for branch in BRANCHES}
    (args.output/'thresholds.json').write_text(json.dumps(thresholds,indent=2,allow_nan=False)+'\n')
    with (args.output/'profiles.npz').open('xb') as f:
        np.savez_compressed(f,accounts=arrays['accounts'],**{b:arrays[b+'_gallery'] for b in BRANCHES})
    report={'status':'complete','plan':plan,'coverage':coverage,'metrics':metrics,
            'calibration_scores_sha256':digest_bytes((args.output/'calibration_scores.npz').read_bytes()),
            'thresholds_sha256':digest_bytes((args.output/'thresholds.json').read_bytes()),
            'profiles_sha256':digest_bytes((args.output/'profiles.npz').read_bytes()),
            'selected_checkpoint_sha256':checkpoint_sha,'global_model_fitted':False,'dev_accessed':False,
            'coverage_caveat':'Rates conditional on observed eligible windows; uninspected metadata exclusions have unknown window denominator.',
            'low_far_caveat':'Two calibration accounts cannot establish population low-FAR guarantees.'}
    (args.output/'calibration_complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
