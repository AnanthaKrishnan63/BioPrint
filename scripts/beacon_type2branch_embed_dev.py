"""Frozen DEV encoder inference after completed paired TRAIN calibration."""
import json
import os
from pathlib import Path
import sys
import time
if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.research-type2branch-deps'))
import numpy as np
from beacon_type2branch_embed import ROOT, PROTOCOL, PROTOCOL_SHA256, verify_hashes, validate_arrays
from type2branch_continuity import sha
from type2branch_reference_smoke import module

PREP=ROOT/'research/benchmarks/beacon_type2branch_dev_features_v1'
TRAIN=ROOT/'research/benchmarks/beacon_type2branch_fusion_train_v1'
OUT=ROOT/'research/benchmarks/beacon_type2branch_dev_embeddings_v1'
FROZEN_SHA='fb612db18d0d2635506df53d470b06cc9b5e4115efdc8731293a44a37367a1fe'


def main():
    if sha(PROTOCOL)!=PROTOCOL_SHA256 or sha(TRAIN/'frozen.json')!=FROZEN_SHA:
        raise ValueError('Frozen protocol or calibrated models changed')
    protocol=json.loads(PROTOCOL.read_text());verify_hashes(protocol)
    frozen=json.loads((TRAIN/'frozen.json').read_text())
    if frozen['status']!='paired_train_fusion_complete':raise ValueError('Completed calibration required')
    verify_hashes(json.loads((TRAIN/'plan.json').read_text()))
    for name,config in frozen['models'].items():
        if sha(TRAIN/(name+'.joblib'))!=config['model_sha256']:raise ValueError('Fusion model changed')
    prepared=json.loads((PREP/'report.json').read_text())
    prep_plan=json.loads((PREP/'plan.json').read_text());verify_hashes(prep_plan)
    if prepared['status']!='paired_dev_features_complete' or prep_plan['protocol_sha256']!=PROTOCOL_SHA256:
        raise ValueError('Completed protocol-bound DEV preparation required')
    if prep_plan['source_sha256'].get(str((TRAIN/'frozen.json').relative_to(ROOT)))!=FROZEN_SHA:
        raise ValueError('DEV preparation not bound to frozen calibration')
    if sha(PREP/'features.npz')!=prepared['features_sha256']:raise ValueError('Prepared input changed')
    manifest=json.loads((ROOT/'datasets/beacon/split_manifest.json').read_text())
    identities=sorted({r['participant_id'] for r in manifest['files'] if r['split']=='dev'})
    pair=json.loads((ROOT/'research/benchmarks/type2branch_length_pair_v1/report.json').read_text())
    checkpoint=Path(pair['selected_checkpoint']['checkpoint'])
    original=json.loads((ROOT/'research/benchmarks/type2branch_train_v1/plan.json').read_text())
    verify_hashes(original)
    sources=[Path(__file__),PROTOCOL,TRAIN/'frozen.json',TRAIN/'plan.json',PREP/'plan.json',PREP/'report.json',
        ROOT/'scripts/beacon_type2branch_embed.py',ROOT/'scripts/type2branch_reference_smoke.py']
    plan={'protocol_sha256':PROTOCOL_SHA256,'frozen_sha256':FROZEN_SHA,
        'source_sha256':{**protocol['source_sha256'],**original['source_sha256'],
            **{str(p.relative_to(ROOT)):sha(p) for p in sources}},
        'input_sha256':{str((PREP/'features.npz').relative_to(ROOT)):prepared['features_sha256']}}
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    with np.load(PREP/'features.npz',allow_pickle=False) as a:x,metadata=validate_arrays(a,identities)
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import tensorflow as tf
    if np.__version__!='1.26.4':raise ValueError('Pinned NumPy required')
    ref=ROOT/'research/benchmarks/references/type2branch';module('conf',ref/'conf.small.1Kusers.py')
    model=module('beacon_dev_model',ref/'model.py').get_model_Type2Branch({'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
    reader=tf.train.load_checkpoint(str(checkpoint))
    if int(reader.get_tensor('optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'))!=500:raise ValueError('Checkpoint iteration mismatch')
    restored=tf.train.Checkpoint(model=model).restore(str(checkpoint));restored.assert_existing_objects_matched();restored.expect_partial()
    started=time.perf_counter()
    embeddings=np.concatenate([model(x[i:i+32],training=False).numpy() for i in range(0,len(x),32)])
    if embeddings.shape!=(len(x),256) or not np.isfinite(embeddings).all():raise ValueError('Invalid encoder output')
    verify_hashes(plan);verify_hashes(prep_plan)
    np.savez_compressed(OUT/'embeddings.npz',embeddings=embeddings,**metadata)
    report={'status':'paired_dev_embeddings_complete','embeddings_sha256':sha(OUT/'embeddings.npz'),
        'features_sha256':prepared['features_sha256'],'rows':len(x),'seconds':time.perf_counter()-started,
        'below25_windows':int((metadata['true_length']<25).sum()),'model_updates':0,'test_observations_decoded':False}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
