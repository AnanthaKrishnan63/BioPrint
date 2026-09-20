"""Prospective same-window FIT-TRAIN tap/encoder comparison; no held-out IO."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hmog_fit_tap_extract import validate_metadata,META
from hmog_tap_benchmark import build_profiles,score_windows
from hmog_score_fusion import fit_scales,fuse
from hmog_select import embed_all,load_checkpoint
from hmog_training_core import load_author_classes
from hmog_verification import mean_gallery,euclidean_scores,far_threshold,verification_rates,pooled_eer

ROOT=Path(__file__).resolve().parents[1]
ACCOUNTS=sorted(['717868','526319','986737','539502'])
BRANCHES=('key','imu','joint')
FUSIONS={'key_imu_tap11':('key','imu','tap11'),'joint_tap11':('joint','tap11')}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def load_paired(feature_path,tap_path):
    with np.load(feature_path,allow_pickle=False) as base, np.load(tap_path,allow_pickle=False) as tap:
        meta=validate_metadata({n:base[n].copy() for n in META})
        # Exact original-window order before copying any biometric features.
        for n in META:
            if not np.array_equal(meta[n],tap['window_'+n]):raise ValueError('Tap/base window mismatch: '+n)
        key,imu=base['key'].copy(),base['imu'].copy()
        vectors,index=tap['tap_vectors'].copy(),tap['tap_window'].copy()
    if key.shape!=(207,50,10) or imu.shape!=(207,100,24) or key.dtype!=np.float32 or imu.dtype!=np.float32:
        raise ValueError('Wrong frozen encoder feature schema')
    if not np.isfinite(key).all() or not np.isfinite(imu).all():raise ValueError('Nonfinite encoder features')
    return meta,key,imu,vectors,index


def split_windows(subjects,sessions,scorable):
    gallery=np.zeros(len(subjects),dtype=bool);calibration=gallery.copy();diagnostic=sessions>=9
    choices={};missing=[]
    for sid in ACCOUNTS:
        eligible=sorted(set(sessions[(subjects==sid)&(sessions<=8)&scorable].tolist()))
        if not eligible:missing.append({'subject':sid,'reason':'no_eligible_enrollment_session'});continue
        first=eligible[0];choices[sid]=first
        gallery|=(subjects==sid)&(sessions==first)&scorable
        calibration|=(subjects==sid)&(sessions<=8)&(sessions!=first)
    return gallery,calibration,diagnostic,choices,missing


def paired_scores(embeddings,meta,taps,index,reference=None):
    from hmog_tap_benchmark import validate_taps
    taps,index=validate_taps(taps,index,len(meta['subject']))
    counts=np.bincount(index,minlength=len(meta['subject']));scorable=counts>=5
    gallery,calibration,diagnostic,choices,missing=split_windows(meta['subject'],meta['session'],scorable)
    if reference is None:
        profiles,profile_diag=build_profiles(taps,index,meta['subject'],gallery,ACCOUNTS)
        missing.extend({'subject':sid,'reason':'missing_tap_gallery'} for sid in ACCOUNTS if sid not in profiles)
        if missing:return {'status':'infeasible','missing':missing,'gallery_sessions':choices,'profile_diagnostics':profile_diag},None
        tap=score_windows(taps,index,len(meta['subject']),profiles,ACCOUNTS)
    else:
        profile_diag=reference['complete']['profile_counts'];tap=reference['scores']
        for name,mask in [('gallery',gallery),('calibration',calibration),('diagnostic',diagnostic)]:
            if not np.array_equal(mask,tap[name]):raise ValueError('Frozen full11 split mismatch')
        if tap['accounts'].astype(str).tolist()!=ACCOUNTS:raise ValueError('Frozen full11 account mismatch')
        if missing:raise ValueError('Unexpected missing shared galleries')
    scores={'tap11':tap['distances']};available={'tap11':tap['available']}
    for b in BRANCHES:
        centers=mean_gallery(embeddings[b][gallery],meta['subject'][gallery])
        accounts,distances=euclidean_scores(embeddings[b],centers)
        if accounts!=ACCOUNTS:raise ValueError('Missing fixed encoder gallery')
        scores[b]=distances;available[b]=np.isfinite(distances)
    genuine=meta['subject'][:,None]==np.array(ACCOUNTS)[None,:]
    scales={}
    for name,branches in FUSIONS.items():
        tensor=np.stack([scores[b] for b in branches],axis=2)
        mask=np.stack([available[b] for b in branches],axis=2)
        scales[name]=fit_scales(tensor,mask,diagnostic[:,None]&~genuine)
        scores[name],available[name]=fuse(tensor,mask,scales[name])
    intersection=np.logical_and.reduce([available[b] for b in scores])
    metrics={};thresholds={};infeasible=[]
    for name,values in scores.items():
        usable=available[name];cal=calibration[:,None]&usable
        cg,ci=values[cal&genuine],values[cal&~genuine]
        if not len(cg) or not len(ci):infeasible.append(name);continue
        thresholds[name]=(reference['complete']['tap11']['thresholds'] if name=='tap11' and reference is not None
                          else {str(t):far_threshold(ci,t) for t in (.001,.01,.05)})
        metrics[name]={}
        for scope,mask in [('base_coverage',usable),('shared_tap_intersection',intersection)]:
            obs=diagnostic[:,None]&mask
            g,i=values[obs&genuine],values[obs&~genuine]
            metrics[name][scope]={'pooled_eer_diagnostic':pooled_eer(g,i) if len(g) and len(i) else None,
              'operating_points':{target:verification_rates(g,i,t,expected_genuine=int(diagnostic.sum()),
                expected_impostor=int(diagnostic.sum())*3) for target,t in thresholds[name].items()}}
    missing_accounts=[sid for sid in ACCOUNTS if not np.any(calibration&scorable&(meta['subject']==sid))
                       or not np.any(diagnostic&scorable&(meta['subject']==sid))]
    status='infeasible' if infeasible or missing_accounts else 'complete'
    result={'status':status,'missing_calibration_or_diagnostic_accounts':missing_accounts,
        'infeasible_methods':infeasible,'gallery_sessions':choices,'profile_diagnostics':profile_diag,
        'scales':{k:v.tolist() for k,v in scales.items()},'thresholds':thresholds,'metrics':metrics,
        'counts':{'base_windows':len(scorable),'tap_scorable_windows':int(scorable.sum()),
            'gallery_windows':int(gallery.sum()),'calibration_base_windows':int(calibration.sum()),
            'diagnostic_base_windows':int(diagnostic.sum())},
        'limitation':'TRAIN diagnostic is optimistic: encoder optimized on sessions9–16 and fusion scales fitted on the same diagnostic impostor claims. No held-out result or population performance claim.'}
    arrays={'accounts':np.array(ACCOUNTS),'genuine':genuine,'gallery_mask':gallery,'calibration_mask':calibration,
            'diagnostic_mask':diagnostic,'shared_intersection':intersection,
            **{'distance_'+k:v for k,v in scores.items()},**{'available_'+k:v for k,v in available.items()}}
    return result,arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    for name in ('features','tap-features','tap-report','tap-diagnostic-dir','selection-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('features','tap_features','tap_report','tap_diagnostic_dir','selection_dir','output'):
        if not getattr(args,name).resolve().is_relative_to(ROOT):raise ValueError('Repository paths only')
    plan_path=args.output/'paired_train_plan.json'
    if args.action=='prepare':
        report=json.loads(args.tap_report.read_text());selection=json.loads((args.selection_dir/'selection_complete.json').read_text())
        if selection.get('status')!='complete' or selection.get('selected_epoch')!=3:raise ValueError('Frozen epoch3 selection required')
        if report.get('feature_archive_sha256')!=digest(args.tap_features) or report.get('held_out_measurements_read')is not False:
            raise ValueError('Tap artifact provenance mismatch')
        standalone=json.loads((args.tap_diagnostic_dir/'complete.json').read_text())
        if standalone.get('status')!='complete_train_diagnostic' or standalone['scores_sha256']!=digest(args.tap_diagnostic_dir/'scores.npz'):
            raise ValueError('Complete frozen full11 diagnostic required')
        args.output.mkdir(parents=True,exist_ok=False)
        inputs={str(p.resolve().relative_to(ROOT)):digest(p) for p in [args.features,args.tap_features,args.tap_report,
            args.selection_dir/'selection_complete.json',args.selection_dir/'selected_checkpoint.pt',
            args.tap_diagnostic_dir/'complete.json',args.tap_diagnostic_dir/'scores.npz',args.tap_diagnostic_dir/'profiles.json',
            ROOT/'research/benchmarks/hmog_tap_protocol_proposal_v1.json',
            ROOT/'research/benchmarks/hmog_tap_implementation_contract_v1.md']}
        sources={n:digest(ROOT/'scripts'/n) for n in ['hmog_tap_paired_train.py','hmog_fit_tap_extract.py',
            'hmog_tap_benchmark.py','hmog_tap_reference.py','hmog_score_fusion.py','hmog_select.py','hmog_verification.py','hmog_training_core.py']}
        plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':inputs,'source_sha256':sources,
            'checkpoint_sha256':selection['selected_checkpoint_sha256'],'accounts':ACCOUNTS,
            'methods':['tap11','key','imu','joint','key_imu_tap11','joint_tap11'],
            'gallery':'First tap-eligible numeric enrollment session peraccount; only windows>=5taps; allmethods samegallerywindowIDs; min80tapvectors',
            'calibration':'Remaining sessions1–8; thresholds empiricalFAR.001/.01/.05',
            'fusion':'Equalweight distances scaled by mean same-complete-case impostordistance in9–16TRAIN only',
            'diagnostic':'Sessions9–16 explicitlyoptimistic encodertraining+fusion-scalefitting data, no held-outclaims',
            'cpu_threads':2,'no_raw_or_heldout_reads':True}
        plan_path.write_text(json.dumps(plan,indent=2))
        (args.output/'source_hmog_tap_paired_train.py').write_bytes(Path(__file__).read_bytes())
        print('Prepared only; no feature arrays observed');return
    plan=json.loads(plan_path.read_text())
    for rel,sha in plan['input_sha256'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('Frozen input changed')
    for n,sha in plan['source_sha256'].items():
        if digest(ROOT/'scripts'/n)!=sha:raise ValueError('Reviewed source changed')
    if (args.output/'paired_train_complete.json').exists():raise ValueError('Preserve existing diagnostic')
    meta,key,imu,taps,index=load_paired(args.features,args.tap_features)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    cls,_=load_author_classes();model=cls(10,24,50,100,64).eval()
    load_checkpoint(model,args.selection_dir/'selected_checkpoint.pt',plan['checkpoint_sha256'])
    embeddings=embed_all(model,key,imu)
    with np.load(args.tap_diagnostic_dir/'scores.npz',allow_pickle=False) as frozen:
        for n in META:
            if not np.array_equal(meta[n],frozen[n]):raise ValueError('Frozen tap diagnostic window mismatch')
        reference={'scores':{n:frozen[n].copy() for n in ('distances','available','gallery','calibration','diagnostic','accounts')},
                   'complete':json.loads((args.tap_diagnostic_dir/'complete.json').read_text())}
    result,arrays=paired_scores(embeddings,meta,taps,index,reference=reference)
    result['plan']=plan
    if arrays is not None:
        with (args.output/'paired_train_scores.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        result['score_archive_sha256']=digest(args.output/'paired_train_scores.npz')
    (args.output/'paired_train_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':result['status']}))

if __name__=='__main__':main()
