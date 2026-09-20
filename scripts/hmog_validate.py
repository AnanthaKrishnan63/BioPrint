"""Frozen HMOG dev validation; no fitting, tuning, or implicit cohort access."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import torch
from hmog_select import read_json,digest_bytes,load_checkpoint,embed_all
from hmog_training_core import load_author_classes
from hmog_verification import mean_gallery,euclidean_scores,pooled_eer,verification_rates

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/benchmarks/hmog/model_protocol_v2.json'
DEV_IDS=frozenset(('556357','219303','777078','737973'))
BRANCHES=('joint','key','imu')
TARGETS=('0.001','0.01','0.05')


def load_dev_arrays(path):
    """Metadata validation must precede copying any key/IMU array."""
    with np.load(path,allow_pickle=False) as archive:
        subjects,sessions,roles=(archive[n].copy() for n in ('subject','session','role'))
        if (subjects.ndim!=1 or subjects.dtype.kind not in 'US' or roles.shape!=subjects.shape
            or roles.dtype.kind not in 'US' or sessions.shape!=subjects.shape or sessions.dtype.kind not in 'iu'):
            raise ValueError('Invalid dev metadata')
        subjects,roles=subjects.astype(str),roles.astype(str)
        if not set(subjects).issubset(DEV_IDS):raise PermissionError('Non-dev identity forbidden')
        for session,role in zip(sessions,roles):
            if not 1<=session<=16 or role!=('train_support' if session<=8 else 'dev_probe'):
                raise PermissionError('Unauthorized dev session/role')
        key,imu=archive['key'].copy(),archive['imu'].copy()
    if (key.shape!=(len(subjects),50,10) or imu.shape!=(len(subjects),100,24)
        or key.dtype!=np.float32 or imu.dtype!=np.float32
        or not np.isfinite(key).all() or not np.isfinite(imu).all()):
        raise ValueError('Invalid finite float32 dev features')
    return key,imu,subjects,sessions,roles


def coverage_from_report(report,subjects,sessions):
    reports=report.get('subjects',{});excluded=report.get('excluded_sessions',{})
    if set(reports)!=DEV_IDS or not set(excluded).issubset(DEV_IDS):raise ValueError('Wrong coverage population')
    result={}
    for sid in sorted(DEV_IDS):
        inspected=reports[sid]; omitted=excluded.get(sid,{})
        if set(inspected)&set(omitted) or set(inspected)|set(omitted)!={str(s) for s in range(1,17)}:
            raise ValueError('All sessions1–16 require inspected or excluded coverage')
        for s,r in omitted.items():
            if (r.get('reason') not in ('missing_keypress','empty_keypress') or r.get('basis')!='archive_metadata'
                or r.get('activity_contents_inspected',False) is not False
                or r.get('candidate_windows') is not None or r.get('eligible_windows') is not None
                or np.any((subjects==sid)&(sessions==int(s)))):
                raise ValueError('Excluded sessions must retain unknown counts and no features')
        result[sid]={}
        for role,nums in [('train_support',range(1,9)),('dev_probe',range(9,17))]:
            candidate=eligible=0
            for s in nums:
                if str(s) not in inspected:continue
                r=inspected[str(s)];c,e=r.get('candidate_windows'),r.get('eligible_windows')
                if type(c)is not int or type(e)is not int or not 0<=e<=c or e!=int(np.sum((subjects==sid)&(sessions==s))):
                    raise ValueError('Coverage mismatch')
                candidate+=c;eligible+=e
            result[sid][role]={'candidate_windows':candidate,'eligible_windows':eligible,
                'missing_windows':candidate-eligible,'coverage':eligible/candidate if candidate else None,
                'denominator_scope':'inspected metadata-nonempty sessions only',
                'excluded_sessions':{str(s):omitted[str(s)] for s in nums if str(s) in omitted},
                'excluded_session_candidate_windows':None}
    return result


def validate_thresholds(thresholds):
    if set(thresholds)!=set(BRANCHES):raise ValueError('Require frozen thresholds for all three branches')
    for b in BRANCHES:
        if set(thresholds[b])!=set(TARGETS):raise ValueError('Require all three frozen targets')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in thresholds[b].values()):
            raise ValueError('Thresholds must be finite numeric values')
    return thresholds


def evaluate_embeddings(embeddings,subjects,roles,coverage,thresholds):
    thresholds=validate_thresholds(thresholds)
    accounts=sorted(DEV_IDS);support=roles=='train_support';probe=roles=='dev_probe'
    labels=subjects[probe];genuine=labels[:,None]==np.array(accounts)[None,:]
    arrays={'accounts':np.array(accounts),'probe_subject':labels,'genuine_mask':genuine}
    missing=[{'subject':sid,'missing_role':role} for sid in accounts for role in ('train_support','dev_probe')
             if not np.any((subjects==sid)&(roles==role))]
    expected={sid:coverage[sid]['dev_probe']['candidate_windows'] for sid in accounts}
    metrics={}
    for b in BRANCHES:
        z=np.asarray(embeddings[b])
        if z.shape!=(len(subjects),64) or not np.isfinite(z).all():raise ValueError('Invalid embedding shape/value')
        gallery=mean_gallery(z[support],subjects[support])
        observed_accounts,observed=euclidean_scores(z[probe],gallery)
        distances=np.full((len(labels),len(accounts)),np.nan)
        profiles=np.full((len(accounts),64),np.nan)
        for column,sid in enumerate(observed_accounts):
            target=accounts.index(sid);distances[:,target]=observed[:,column];profiles[target]=gallery[sid]
        available=np.isfinite(distances)
        arrays[b+'_distances']=distances;arrays[b+'_profiles']=profiles
        arrays[b+'_claim_available']=available
        arrays[b+'_embeddings']=z
        g=distances[genuine&available];i=distances[~genuine&available]
        branch={'pooled_eer_diagnostic':pooled_eer(g,i) if len(g) and len(i) else None,
                'operating_points':{},'per_account':{},'missing_gallery_accounts':[s for s in accounts if s not in gallery]}
        for target,t in thresholds[b].items():
            branch['operating_points'][target]=verification_rates(g,i,t,
                expected_genuine=sum(expected.values()),expected_impostor=sum(expected.values())*(len(accounts)-1))
            arrays[b+'_accepted_'+target]=available&(distances<=t)
        for c,sid in enumerate(accounts):
            mask=labels==sid; valid=available[:,c]
            gg=distances[mask&valid,c];ii=distances[~mask&valid,c]
            branch['per_account'][sid]={'gallery_available':sid in gallery,
                'probe_windows':int(mask.sum()),'eer_diagnostic':pooled_eer(gg,ii) if len(gg) and len(ii) else None,
                'operating_points':{target:verification_rates(gg,ii,t,expected_genuine=expected[sid],
                    expected_impostor=sum(v for k,v in expected.items() if k!=sid)) for target,t in thresholds[b].items()}}
        metrics[b]=branch
    return {'status':'infeasible' if missing else 'complete','missing':missing,'metrics':metrics},arrays


def main():
    parser=argparse.ArgumentParser()
    for n in ('selection-dir','calibration-dir','features','extraction-report','output'):
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args()
    for p in vars(args).values():
        if not p.resolve().is_relative_to(ROOT):raise ValueError('Repository paths only')
    protocol,protocol_sha=read_json(PROTOCOL)
    selection,selection_sha=read_json(args.selection_dir/'selection_complete.json')
    calibration,calibration_sha=read_json(args.calibration_dir/'calibration_complete.json')
    extraction,extraction_sha=read_json(args.extraction_report)
    if protocol.get('version')!=2 or selection.get('status')!='complete' or calibration.get('status')!='complete':
        raise ValueError('Complete frozen selection/calibration required')
    checkpoint_sha=selection['selected_checkpoint_sha256']
    if selection['plan']['model_protocol_sha256']!=protocol_sha:raise ValueError('Selection protocol mismatch')
    # Calibration binding/schema is validated before any dev feature access.
    if (calibration['plan']['model_protocol_sha256']!=protocol_sha
        or calibration['plan']['selected_checkpoint_sha256']!=checkpoint_sha
        or calibration['plan']['selection_report_sha256']!=selection_sha):
        raise ValueError('Calibration checkpoint/protocol mismatch')
    threshold_document,threshold_sha=read_json(args.calibration_dir/'thresholds.json')
    if calibration.get('thresholds_sha256')!=threshold_sha:raise ValueError('Frozen threshold hash mismatch')
    thresholds=validate_thresholds(threshold_document)
    feature_sha=digest_bytes(args.features.read_bytes())
    if extraction.get('feature_archive_sha256')!=feature_sha:raise ValueError('Dev feature archive/report mismatch')
    if extraction['plan'].get('model_protocol_sha256')!=protocol_sha:raise ValueError('Dev extractor protocol mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    sources={n:(ROOT/'scripts'/n).read_bytes() for n in ['hmog_validate.py','hmog_select.py','hmog_verification.py','hmog_training_core.py']}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'dev_ids':sorted(DEV_IDS),
        'model_protocol_sha256':protocol_sha,'selection_complete_sha256':selection_sha,
        'calibration_complete_sha256':calibration_sha,'selected_checkpoint_sha256':checkpoint_sha,
        'thresholds_sha256':threshold_sha,
        'frozen_thresholds':thresholds,'features_sha256':feature_sha,'extraction_report_sha256':extraction_sha,
        'sources_sha256':{n:digest_bytes(v) for n,v in sources.items()},'gallery':'supportonly arithmeticmean64D',
        'decision':'distance<=frozen threshold; unavailable claims abstain',
        'no_fitting_or_tuning':True,'test_measurements_read':False}
    calibration_provenance={'plan_fields':{},'report_fields':{}}
    for field in ('calibration_protocol_sha256','calibration_scope','limitation','limitations'):
        if field in calibration.get('plan',{}):
            calibration_provenance['plan_fields'][field]=calibration['plan'][field]
            calibration_provenance[field]=calibration['plan'][field]
        if field in calibration:
            calibration_provenance['report_fields'][field]=calibration[field]
            calibration_provenance[field]=calibration[field]
    plan['calibration_provenance']=calibration_provenance
    if 'calibration_protocol_sha256' in calibration_provenance:
        plan['calibration_protocol_sha256']=calibration_provenance['calibration_protocol_sha256']
    (args.output/'validation_plan.json').write_text(json.dumps(plan,indent=2))
    for n,raw in sources.items():(args.output/('source_'+n)).write_bytes(raw)
    key,imu,subjects,sessions,roles=load_dev_arrays(args.features)
    if digest_bytes(args.features.read_bytes())!=feature_sha:raise ValueError('Feature archive changed')
    coverage=coverage_from_report(extraction,subjects,sessions)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    model_class,_=load_author_classes();model=model_class(10,24,50,100,64).eval()
    load_checkpoint(model,args.selection_dir/'selected_checkpoint.pt',checkpoint_sha)
    embeddings=embed_all(model,key,imu) if len(key) else {b:np.empty((0,64)) for b in BRANCHES}
    result,arrays=evaluate_embeddings(embeddings,subjects,roles,coverage,thresholds)
    arrays.update({'key':key,'imu':imu,'subject':subjects,'session':sessions,'role':roles})
    with (args.output/'dev_replay.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    with (args.output/'profiles.npz').open('xb') as f:
        np.savez_compressed(f,accounts=arrays['accounts'],**{b:arrays[b+'_profiles'] for b in BRANCHES})
    probe=roles=='dev_probe'
    with (args.output/'dev_features.npz').open('xb') as f:
        np.savez_compressed(f,key=key[probe],imu=imu[probe],subject=subjects[probe],
                            session=sessions[probe],role=roles[probe])
    result.update({'plan':plan,'coverage':coverage,'replay_sha256':digest_bytes((args.output/'dev_replay.npz').read_bytes()),
        'calibration_provenance':calibration_provenance,
        'profiles_sha256':digest_bytes((args.output/'profiles.npz').read_bytes()),
        'dev_features_sha256':digest_bytes((args.output/'dev_features.npz').read_bytes()),
        'validation_only':True,'test_measurements_read':False,'production_database_used':False,'listener_started':False})
    (args.output/'validation_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':result['status'],'probe_windows':int(np.sum(roles=='dev_probe'))}))

if __name__=='__main__':main()
