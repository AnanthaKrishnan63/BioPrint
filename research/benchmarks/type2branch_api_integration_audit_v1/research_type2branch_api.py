"""Frozen DEV replay only; parent research app supplies the loopback gate.

TensorFlow inference belongs to the isolated worker, never this process.
Cached artifacts are immutable for a pinned release; restart after updates.
"""
from functools import lru_cache
import io
import json
import threading

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
import numpy as np

from type2branch_release import RELEASE, digest, verify_release
from type2branch_wire import validate_features

MANIFEST_SHA256 = None
router = APIRouter(prefix='/api/type2branch', tags=['Type2Branch research'])
_worker = None
_worker_pin = None
_worker_lock = threading.Lock()


@lru_cache(maxsize=1)
def _artifacts(pin):
    manifest = verify_release(pin)
    blobs = {name: (RELEASE / name).read_bytes()
             for name in ('dev_features.npz', 'dev_report.json')}
    if any(digest(blob) != manifest['files'][name] for name, blob in blobs.items()):
        raise ValueError('Release changed while loading')
    accounts = manifest['subjects']
    if (not isinstance(accounts, list) or len(accounts) != 79
            or not all(isinstance(s, str) for s in accounts)
            or accounts != sorted(set(accounts))):
        raise ValueError('Expected79 distinct ordered accounts')
    threshold = manifest['threshold']
    if type(threshold) not in (int, float) or not np.isfinite(threshold):
        raise ValueError('Finite frozen threshold required')
    with np.load(io.BytesIO(blobs['dev_features.npz']), allow_pickle=False) as archive:
        features, subjects, windows = (archive[k].copy() for k in ('features', 'subject', 'window'))
    if (features.shape != (790, 100, 5) or features.dtype != np.float32
            or subjects.shape != (790,) or subjects.dtype.kind not in 'US'
            or windows.shape != (790,) or windows.dtype.kind not in 'iu'):
        raise ValueError('Unexpected DEV feature schema')
    subjects = subjects.astype(str)
    if set(subjects) != set(accounts):
        raise ValueError('DEV accounts differ from release')
    for account in accounts:
        if not np.array_equal(np.sort(windows[subjects == account]), np.arange(10)):
            raise ValueError('Expected exactly ten DEV probes per account')
    for start in range(0, len(features), 32):
        validate_features(features[start:start + 32])
    report = json.loads(blobs['dev_report.json'])
    if not isinstance(report, dict):
        raise ValueError('Expected frozen report object')
    return {'features': features, 'subjects': subjects, 'windows': windows,
            'accounts': accounts, 'threshold': float(threshold), 'report': report}


def artifacts():
    try:
        return _artifacts(MANIFEST_SHA256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(503, 'Type2Branch release unavailable or failed integrity validation') from error


def get_worker():
    global _worker, _worker_pin
    if MANIFEST_SHA256 is None:
        raise HTTPException(503, 'Type2Branch release is not frozen')
    with _worker_lock:
        if _worker is None or _worker_pin != MANIFEST_SHA256:
            if _worker is not None:
                _worker.close()
                _worker = None
                _worker_pin = None
            from type2branch_worker_client import WorkerClient
            _worker = WorkerClient(MANIFEST_SHA256)
            _worker_pin = MANIFEST_SHA256
        return _worker


def close_worker():
    global _worker, _worker_pin
    with _worker_lock:
        if _worker is not None:
            try:
                _worker.close()
            finally:
                _worker = None
                _worker_pin = None


@router.get('/{split}/samples')
async def samples(split: str, offset: int = Query(0, ge=0), limit: int = Query(32, ge=1, le=32)):
    if split != 'dev':
        raise HTTPException(403, 'Only frozen DEV probes are exposed; test remains sealed')
    data = artifacts()
    end = offset + limit
    return {'split': 'dev', 'total': len(data['features']), 'offset': offset,
            'features': data['features'][offset:end].tolist(),
            'subjects': data['subjects'][offset:end].tolist(),
            'windows': data['windows'][offset:end].tolist(), 'accounts': data['accounts']}


@router.get('/results')
async def results():
    return artifacts()['report']


@router.post('/score')
async def score(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'features'}:
            raise ValueError('Expected exactly features')
        features = validate_features(body['features'])
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(422, 'Expected1..32 valid100x5 Type2Branch feature windows') from error
    data = artifacts()
    try:
        def inference():
            return get_worker().score(features)
        output = await run_in_threadpool(inference)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        raise HTTPException(503, 'Type2Branch inference worker unavailable') from error
    return {'accounts': data['accounts'], 'threshold': data['threshold'],
            'scores': output['scores'], 'accepted': output['accepted'],
            'score_direction': 'larger_is_genuine',
            'threshold_source': 'frozen inner-TRAIN calibration'}
