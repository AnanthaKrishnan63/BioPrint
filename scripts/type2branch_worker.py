"""Persistent local stdin/stdout inference worker; fixed hash-pinned DEV release."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES=2_000_000


def send(value):
    print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-sha',required=True)
    args=parser.parse_args()
    from type2branch_release import RELEASE, verify_release
    manifest=verify_release(args.manifest_sha)
    os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='',TF_NUM_INTEROP_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2',
        KERAS_HOME=str(ROOT/'.research-tmp/type2branch-keras'),
        XDG_CACHE_HOME=str(ROOT/'.research-tmp/type2branch-cache'))
    sys.path.insert(0,str(ROOT/'.research-type2branch-deps'))
    import numpy as np
    import tensorflow as tf
    from type2branch_reference_smoke import module
    from type2branch_wire import validate_request
    from type2branch_scoring import gallery_scores, accept_scores
    if np.__version__!='1.26.4':raise ValueError('Pinned NumPy required')
    with np.load(io.BytesIO((RELEASE/'profiles.npz').read_bytes()),allow_pickle=False) as a:
        gallery=a['gallery'];subjects=a['subjects'];threshold=a['threshold']
    if gallery.shape!=(79,5,256) or subjects.shape!=(79,) or threshold.ndim!=0:
        raise ValueError('Frozen profile shape mismatch')
    if len(set(subjects))!=79 or not np.isfinite(gallery).all() or not np.isfinite(threshold):
        raise ValueError('Invalid profiles')
    if subjects.tolist()!=manifest['subjects'] or float(threshold)!=manifest['threshold']:
        raise ValueError('Profile metadata mismatch')
    ref=ROOT/'research/benchmarks/references/type2branch'
    with contextlib.redirect_stdout(sys.stderr):
        module('conf',ref/'conf.small.1Kusers.py')
        model=module('type2branch_worker_model',ref/'model.py').get_model_Type2Branch(
            {'SEQUENCE_LENGTH':100,'INPUT_FEATURES':5})['model']
        prefix=str(RELEASE/'encoder')
        reader=tf.train.load_checkpoint(prefix)
        if int(reader.get_tensor('optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'))!=manifest['checkpoint_updates']:
            raise ValueError('Checkpoint iteration mismatch')
        status=tf.train.Checkpoint(model=model).restore(prefix)
        status.assert_existing_objects_matched();status.expect_partial()
    send({'kind':'ready','subjects':subjects.tolist(),'threshold':float(threshold),'manifest_sha':args.manifest_sha})
    while True:
        raw=sys.stdin.buffer.readline(MAX_REQUEST_BYTES+1)
        if not raw:break
        if len(raw)>MAX_REQUEST_BYTES or not raw.endswith(b'\n'):
            raise ValueError('Oversized or incomplete protocol line')
        identifier=None
        try:
            request=json.loads(raw)
            if isinstance(request,dict):identifier=request.get('id')
            identifier,features=validate_request(request)
        except (ValueError,TypeError,OverflowError):
            send({'kind':'error','id':identifier if type(identifier) is int else None,'status':422})
            continue
        with contextlib.redirect_stdout(sys.stderr):
            embeddings=model(features,training=False).numpy()
        if embeddings.shape!=(len(features),256) or not np.isfinite(embeddings).all():
            raise ValueError('Invalid inference output')
        scores=gallery_scores(embeddings,gallery)
        send({'kind':'scores','id':identifier,'scores':scores.tolist(),
              'accepted':accept_scores(scores,float(threshold)).tolist()})


if __name__=='__main__':
    main()
