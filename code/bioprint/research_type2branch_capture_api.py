"""Frozen capture replay; parent research app enforces localhost-only access."""
from functools import lru_cache
import io
import json
import threading

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
import numpy as np

from type2branch_capture_release import RELEASE, digest, verify_release
from type2branch_capture_wire import validate_request

MANIFEST_SHA256 = '58f3fb192a5983eb4fa198de0f432813e1941d259429b557acf9b57ad8049657'
router = APIRouter(prefix='/api/type2branch-capture', tags=['Type2Branch capture research'])
_worker = None
_worker_pin = None
_worker_lock = threading.Lock()


@lru_cache(maxsize=1)
def _artifacts(pin):
    manifest = verify_release(pin)
    blobs = {name: (RELEASE / name).read_bytes() for name in ('dev_features.npz', 'dev_report.json')}
    if any(digest(blob) != manifest['files'][name] for name, blob in blobs.items()):
        raise ValueError('Release changed while loading')
    accounts, thresholds = manifest['subjects'], manifest['thresholds']
    if (not isinstance(accounts, list) or len(accounts) != 79
            or not all(isinstance(s, str) and s for s in accounts)
            or accounts != sorted(set(accounts))):
        raise ValueError('Expected79 distinct ordered accounts')
    if (not isinstance(thresholds, dict) or set(thresholds) != {'25', '50', '75', '100'}
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   for value in thresholds.values())):
        raise ValueError('Four finite calibrated thresholds required')
    count = manifest['eligible_probes']
    if type(count) is not int or not 0 <= count <= 79:
        raise ValueError('Invalid eligible capture count')
    report = json.loads(blobs['dev_report.json'])
    if report.get('status') != 'frozen_capture_dev_evaluation_complete':
        raise ValueError('Completed capture evaluation required')
    coverage = report['metrics']['coverage']
    if (type(coverage['eligible']) is not int or coverage['eligible'] != count
            or type(coverage['total']) is not int or coverage['total'] != 79
            or type(coverage['ineligible_count']) is not int or coverage['ineligible_count'] != 79 - count):
        raise ValueError('Manifest/report capture coverage mismatch')
    ineligible = coverage['ineligible']
    if (not isinstance(ineligible, list) or len(ineligible) != 79 - count
            or any(not isinstance(row, dict) or not isinstance(row.get('identity'), str)
                   or not isinstance(row.get('reason'), str) or not row['reason'] for row in ineligible)):
        raise ValueError('Invalid unavailable-capture ledger')
    missing = [row['identity'] for row in ineligible]
    if len(set(missing)) != len(missing) or not set(missing).issubset(accounts):
        raise ValueError('Unavailable identities differ from accounts')
    with np.load(io.BytesIO(blobs['dev_features.npz']), allow_pickle=False) as archive:
        features, subjects, windows, lengths = (archive[k].copy()
            for k in ('features', 'subject', 'window', 'true_length'))
    if (features.shape != (count, 100, 5) or features.dtype != np.float32
            or subjects.shape != (count,) or subjects.dtype.kind not in 'US'
            or windows.shape != (count,) or windows.dtype.kind not in 'iu' or np.any(windows != 0)
            or lengths.shape != (count,) or lengths.dtype.kind not in 'iu'):
        raise ValueError('Unexpected capture feature schema')
    subjects = subjects.astype(str)
    if len(set(subjects)) != count or set(subjects) != set(accounts) - set(missing):
        raise ValueError('Expected exactly one probe per eligible account')
    for start in range(0, count, 32):
        validate_request({'id': 0, 'features': features[start:start + 32],
                          'true_lengths': lengths[start:start + 32].tolist()})
    for length in (25, 50, 75, 100):
        expected = int(np.sum(lengths == length))
        actual = coverage['by_length'][str(length)]['eligible']
        if type(actual) is not int or actual != expected:
            raise ValueError('Reported length coverage differs from features')
    if count == 0 and report['metrics']['conditional_metrics'] != {}:
        raise ValueError('Unavailable captures must not have conditional score metrics')
    return {'features': features, 'subjects': subjects, 'windows': windows,
            'true_lengths': lengths, 'accounts': accounts, 'thresholds': thresholds, 'report': report}


def artifacts():
    try:
        return _artifacts(MANIFEST_SHA256)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise HTTPException(503, 'Capture release unavailable or failed integrity validation') from error


def get_worker():
    global _worker, _worker_pin
    if MANIFEST_SHA256 is None:
        raise HTTPException(503, 'Capture release is not frozen')
    with _worker_lock:
        if _worker is None or _worker_pin != MANIFEST_SHA256:
            if _worker is not None:
                _worker.close()
                _worker = None
                _worker_pin = None
            from type2branch_capture_worker_client import WorkerClient
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
        raise HTTPException(403, 'Only frozen DEV captures are exposed; test remains sealed')
    data = artifacts()
    end = offset + limit
    return {'split': 'dev', 'total': len(data['features']), 'offset': offset,
            'features': data['features'][offset:end].tolist(),
            'subjects': data['subjects'][offset:end].tolist(),
            'windows': data['windows'][offset:end].tolist(),
            'true_lengths': data['true_lengths'][offset:end].tolist(),
            'accounts': data['accounts']}


@router.get('/results')
async def results():
    return artifacts()['report']


@router.post('/score')
async def score(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'features', 'true_lengths'}:
            raise ValueError('Expected exactly features and true_lengths')
        _, features, lengths = validate_request({'id': 0, **body})
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(422, 'Valid padded captures and explicit calibrated lengths required') from error
    data = artifacts()
    try:
        def inference():
            return get_worker().score(features, lengths.tolist())
        output = await run_in_threadpool(inference)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        raise HTTPException(503, 'Capture inference worker unavailable') from error
    return {'accounts': data['accounts'], 'thresholds': data['thresholds'],
            'true_lengths': lengths.tolist(), 'scores': output['scores'], 'accepted': output['accepted'],
            'score_direction': 'larger_is_genuine', 'threshold_source': 'frozen inner-TRAIN exact-length calibration'}
