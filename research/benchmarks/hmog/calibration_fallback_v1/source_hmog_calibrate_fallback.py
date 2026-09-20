"""Separately preregistered calibration on fit identities' unused enrollment sessions."""
import json
from pathlib import Path
import zipfile
import numpy as np
from hmog_select import load_checkpoint,embed_all,BRANCHES,digest_bytes
from hmog_training_core import load_author_classes,FIT_IDS
from hmog_verification import mean_gallery,euclidean_scores,pooled_eer,far_threshold,verification_rates
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/benchmarks/hmog/calibration_fallback_protocol_v1.json'
PROTOCOL_SHA='61a722a2d4d4aeaddc3905adface23529869b0f55936f22ddf30fdae6607eef0'
OUT=ROOT/'research/benchmarks/hmog/calibration_fallback_v1'
TARGETS=(.001,.01,.05)


def partition(subjects,sessions,roles):
    if (subjects.ndim!=1 or subjects.dtype.kind not in 'US' or sessions.shape!=subjects.shape
            or sessions.dtype.kind not in 'iu' or roles.shape!=subjects.shape or roles.dtype.kind not in 'US'):
        raise ValueError('Invalid metadata')
    if set(subjects)!=FIT_IDS:raise PermissionError('Exactly all four fit identities required')
    for session,role in zip(sessions,roles):
        if not 1<=session<=16 or role!=('train_enrollment' if session<=8 else 'train_fit'):
            raise PermissionError('Invalid fit record role/session')
    selected=np.flatnonzero(sessions<=8);labels=[];audit={}
    for sid in sorted(FIT_IDS):
        eligible=sorted(set(int(s) for s in sessions[(subjects==sid)&(sessions<=8)]))
        if len(eligible)<2:raise ValueError('All identities require two enrollment sessions')
        audit[sid]={'gallery_sessions':eligible[:1],'calibration_sessions':eligible[1:],
                    'gallery_windows':int(np.sum((subjects==sid)&(sessions==eligible[0]))),
                    'calibration_windows':int(np.sum((subjects==sid)&(sessions<=8)&(sessions!=eligible[0])))}
    for i in selected:labels.append('gallery' if sessions[i]==audit[str(subjects[i])]['gallery_sessions'][0] else 'calibration')
    return selected,np.array(labels),audit


def selected_feature_rows(path,name,indices,expected_n,tail):
    """Decode only selected NPY rows; excluded train_fit rows stay opaque bytes."""
    wanted=set(int(i) for i in indices);values=[]
    with zipfile.ZipFile(path) as z,z.open(name+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Unsupported NPY header')
        if shape!=(expected_n,*tail) or fortran or dtype!=np.dtype('float32'):raise ValueError('Unexpected feature schema')
        rowbytes=int(np.prod(tail))*dtype.itemsize
        for i in range(expected_n):
            raw=f.read(rowbytes)
            if len(raw)!=rowbytes:raise ValueError('Truncated feature row')
            if i in wanted:
                row=np.frombuffer(raw,dtype=dtype).reshape(tail).copy()
                if not np.isfinite(row).all():raise ValueError('Nonfinite selected feature')
                values.append(row)
    return np.asarray(values,dtype=np.float32)


def score(embeddings,subjects,labels):
    if set(subjects)!=FIT_IDS or set(labels)!={'gallery','calibration'}:raise ValueError('Invalid fallback cohort/roles')
    gallery_mask=labels=='gallery';probe=labels=='calibration'
    for sid in FIT_IDS:
        if not np.any((subjects==sid)&gallery_mask) or not np.any((subjects==sid)&probe):raise ValueError('Missing account gallery/probe')
    accounts=np.array(sorted(FIT_IDS));mask=subjects[probe,None]==accounts[None,:]
    arrays={'accounts':accounts,'probe_subject':subjects[probe],'genuine_mask':mask};metrics={}
    for b in BRANCHES:
        a=np.asarray(embeddings[b])
        if a.shape!=(len(subjects),64) or not np.isfinite(a).all():raise ValueError('Invalid embeddings')
        gallery=mean_gallery(a[gallery_mask],subjects[gallery_mask]);names,dist=euclidean_scores(a[probe],gallery)
        assert names==accounts.tolist()
        g,i=dist[mask],dist[~mask];ops={}
        for target in TARGETS:
            threshold=far_threshold(i,target)
            ops[str(target)]={'threshold':threshold,**verification_rates(g,i,threshold),
                'per_account':{sid:verification_rates(dist[subjects[probe]==sid,j],dist[subjects[probe]!=sid,j],threshold) for j,sid in enumerate(names)}}
        metrics[b]={'pooled_eer_diagnostic':pooled_eer(g,i),'genuine_count':len(g),'impostor_count':len(i),'empirical_far_resolution':1/len(i),'operating_points':ops}
        arrays[b+'_distances']=dist;arrays[b+'_gallery']=np.array([gallery[s] for s in names])
    return metrics,arrays


def main():
    raw=PROTOCOL.read_bytes()
    if digest_bytes(raw)!=PROTOCOL_SHA:raise ValueError('Fallback protocol changed')
    protocol=json.loads(raw);files=protocol['source_files']
    for info in files.values():
        if digest_bytes((ROOT/info['path']).read_bytes())!=info['sha256']:raise ValueError('Frozen source checksum mismatch')
    OUT.mkdir(exist_ok=False)
    feature_path=ROOT/files['fit_features']['path']
    with np.load(feature_path,allow_pickle=False) as z:s,u,r=(z[k].copy() for k in ('subject','session','role'))
    selected,labels,audit=partition(s,u,r)
    sources={n:(ROOT/'scripts'/n).read_bytes() for n in ['hmog_calibrate_fallback.py','hmog_select.py','hmog_verification.py','hmog_training_core.py']}
    plan={'calibration_protocol_sha256':PROTOCOL_SHA,'model_protocol_sha256':files['model_protocol']['sha256'],
          'selection_report_sha256':files['selection_report']['sha256'],'selected_checkpoint_sha256':files['selected_checkpoint']['sha256'],
          'features_sha256':files['fit_features']['sha256'],'failed_calibration_sha256':files['failed_calibration']['sha256'],
          'session_audit':audit,'selected_rows':selected.tolist(),'excluded_train_fit_rows':int(np.sum(u>=9)),
          'sources_sha256':{n:digest_bytes(v) for n,v in sources.items()},'calibration_override_only':True,
          'calibration_scope':'encoder-trained fit identities; disjoint unused enrollment sessions1–8 only',
          'limitation':protocol['caveat'],'caveat':protocol['caveat'],'dev_or_test_accessed':False}
    (OUT/'calibration_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n,v in sources.items():(OUT/('source_'+n)).write_bytes(v)
    key=selected_feature_rows(feature_path,'key',selected,len(s),(50,10))
    imu=selected_feature_rows(feature_path,'imu',selected,len(s),(100,24))
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20260920)
    cls,_=load_author_classes();model=cls(10,24,50,100,64).eval()
    load_checkpoint(model,ROOT/files['selected_checkpoint']['path'],files['selected_checkpoint']['sha256'])
    metrics,arrays=score(embed_all(model,key,imu),s[selected],labels)
    arrays['source_sessions']=u[selected];arrays['source_subjects']=s[selected];arrays['source_roles']=r[selected];arrays['calibration_roles']=labels
    with (OUT/'calibration_scores.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    with (OUT/'profiles.npz').open('xb') as f:np.savez_compressed(f,accounts=arrays['accounts'],**{b:arrays[b+'_gallery'] for b in BRANCHES})
    thresholds={b:{str(t):metrics[b]['operating_points'][str(t)]['threshold'] for t in TARGETS} for b in BRANCHES}
    (OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2,allow_nan=False)+'\n')
    for info in files.values():
        if digest_bytes((ROOT/info['path']).read_bytes())!=info['sha256']:raise ValueError('Frozen source changed during execution')
    report={'status':'complete','plan':plan,'metrics':metrics,'coverage':audit,'calibration_protocol_sha256':PROTOCOL_SHA,
            'selected_checkpoint_sha256':files['selected_checkpoint']['sha256'],'global_model_fitted':False,'dev_accessed':False,
            'calibration_cohort':'encoder-trained fit identities; disjoint unused enrollment sessions only',
            'calibration_scope':plan['calibration_scope'],'limitation':protocol['caveat'],
            'caveat':protocol['caveat'],**{n+'_sha256':digest_bytes((OUT/p).read_bytes()) for n,p in [('thresholds','thresholds.json'),('profiles','profiles.npz'),('calibration_scores','calibration_scores.npz')]}}
    (OUT/'calibration_complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','session_audit':audit,'metrics':metrics,'report_sha256':digest_bytes((OUT/'calibration_complete.json').read_bytes())},indent=2))

if __name__=='__main__':main()
