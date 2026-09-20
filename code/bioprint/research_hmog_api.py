"""Frozen HMOG replay router; parent app restricts access to loopback.

No production database or raw archive reads. Release is unavailable until its
manifest hash is explicitly pinned after validation. Cache checks happen once
per process; restart after any authorized artifact update.
"""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'research/benchmarks/hmog/release_v1'
MANIFEST_SHA256 = '5df75fa341a6b5c0706287f87488a02c2a36ba5185fe2c0033c62a1dc210a111'
BRANCHES = ('joint', 'key', 'imu')
TARGETS = ('0.001', '0.01', '0.05')
FILES = frozenset(('encoder.pt', 'profiles.npz', 'thresholds.json',
                   'dev_features.npz', 'dev_report.json'))
SOURCES = frozenset(('hmog_training_core.py', 'hmog_verification.py'))
router = APIRouter(prefix='/api/hmog', tags=['paired mobile research'])


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def input_arrays(key, imu):
    try:
        key, imu = np.asarray(key, dtype=np.float64), np.asarray(imu, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, 'Rectangular numeric arrays required') from error
    if (key.ndim != 3 or not 1 <= len(key) <= 8 or key.shape[1:] != (50, 10)
            or imu.shape != (len(key), 100, 24)
            or not np.isfinite(key).all() or not np.isfinite(imu).all()
            or np.max(np.abs(key)) > np.finfo(np.float32).max
            or np.max(np.abs(imu)) > np.finfo(np.float32).max):
        raise HTTPException(422, 'Expected 1–8 finite float32-representable paired windows')
    return key.astype(np.float32), imu.astype(np.float32)


@lru_cache(maxsize=1)
def artifacts():
    if MANIFEST_SHA256 is None:
        raise HTTPException(503, 'HMOG validation release is not frozen')
    try:
        raw = (RELEASE / 'manifest.json').read_bytes()
        if digest(raw) != MANIFEST_SHA256:
            raise ValueError('Manifest checksum mismatch')
        manifest = json.loads(raw)
        if (set(manifest['files']) != FILES or set(manifest['sources']) != SOURCES
                or manifest['split'] != 'dev' or manifest['version'] != 1):
            raise ValueError('Release schema mismatch')
        # Validate every declared file before any checkpoint deserialization.
        blobs = {name: (RELEASE / name).read_bytes() for name in FILES}
        for name, blob in blobs.items():
            if digest(blob) != manifest['files'][name]:
                raise ValueError('Artifact checksum mismatch')
        for name in SOURCES:
            if digest((ROOT / 'scripts' / name).read_bytes()) != manifest['sources'][name]:
                raise ValueError('Inference source checksum mismatch')
        dev_report = json.loads(blobs['dev_report.json'])
        with np.load(io.BytesIO(blobs['profiles.npz']), allow_pickle=False) as archive:
            accounts = archive['accounts'].astype(str).tolist()
            profiles = {b: archive[b].copy() for b in BRANCHES}
        if (accounts != sorted(('556357', '219303', '777078', '737973'))
                or any(v.shape != (4, 64) or not np.isfinite(v).all() for v in profiles.values())):
            raise ValueError('Profile account/shape mismatch')
        thresholds = json.loads(blobs['thresholds.json'])
        if set(thresholds) != set(BRANCHES):
            raise ValueError('Threshold branch mismatch')
        for branch in BRANCHES:
            if (set(thresholds[branch]) != set(TARGETS)
                    or not all(type(v) in (float, int) and np.isfinite(v)
                               for v in thresholds[branch].values())):
                raise ValueError('Invalid frozen thresholds')
        with np.load(io.BytesIO(blobs['dev_features.npz']), allow_pickle=False) as archive:
            subjects, sessions, roles = (archive[name].copy() for name in ('subject', 'session', 'role'))
            if (subjects.ndim != 1 or subjects.dtype.kind not in 'US'
                    or sessions.shape != subjects.shape or sessions.dtype.kind not in 'iu'
                    or roles.shape != subjects.shape or roles.dtype.kind not in 'US'
                    or not set(subjects.astype(str)).issubset(accounts)
                    or not np.all((sessions >= 9) & (sessions <= 16))
                    or not np.all(roles.astype(str) == 'dev_probe')):
                raise ValueError('Only designated DEV-probe rows may be exposed')
            key, imu, subjects = archive['key'].copy(), archive['imu'].copy(), archive['subject'].astype(str)
        if (key.ndim != 3 or key.shape[1:] != (50, 10)
                or imu.shape != (len(key), 100, 24) or subjects.shape != (len(key),)
                or not set(subjects).issubset(accounts)
                or key.dtype != np.float32 or imu.dtype != np.float32
                or not np.isfinite(key).all() or not np.isfinite(imu).all()):
            raise ValueError('Invalid frozen DEV observations')
        import torch
        from hmog_training_core import load_author_classes
        model_class, _ = load_author_classes()
        torch.set_num_threads(2)
        model = model_class(10, 24, 50, 100, 64)
        state = torch.load(io.BytesIO(blobs['encoder.pt']), weights_only=True, map_location='cpu')
        if not isinstance(state, dict) or any(not isinstance(x, torch.Tensor) or not torch.isfinite(x).all() for x in state.values()):
            raise ValueError('Nonfinite checkpoint')
        model.load_state_dict(state, strict=True)
        model.eval()
        return {'accounts': accounts, 'profiles': profiles, 'thresholds': thresholds,
                'model': model, 'key': key, 'imu': imu, 'subjects': subjects,
                'validation_status': dev_report.get('status'), 'missing': dev_report.get('missing', []),
                'coverage': dev_report.get('coverage', {})}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise HTTPException(503, 'HMOG release failed integrity validation') from error


class PairedWindows(BaseModel):
    key: list[list[list[float]]] = Field(min_length=1, max_length=8)
    imu: list[list[list[float]]] = Field(min_length=1, max_length=8)


@router.get('/{split}/samples')
async def samples(split: str, offset: int = Query(0, ge=0), limit: int = Query(8, ge=1, le=8)):
    if split != 'dev':
        raise HTTPException(403, 'Only frozen DEV replay is exposed; test remains sealed')
    data = artifacts()
    end = offset + limit
    return {'split': 'dev', 'total': len(data['key']), 'offset': offset,
            'key': data['key'][offset:end].tolist(), 'imu': data['imu'][offset:end].tolist(),
            'subjects': data['subjects'][offset:end].tolist(), 'accounts': data['accounts'],
            'validation_status': data.get('validation_status'), 'missing': data.get('missing', []),
            'coverage': data.get('coverage', {})}


@router.post('/score')
async def score(body: PairedWindows):
    key, imu = input_arrays(body.key, body.imu)
    data = artifacts()
    import torch
    from hmog_verification import branch_embeddings, euclidean_scores
    try:
        outputs = branch_embeddings(data['model'], torch.from_numpy(key), torch.from_numpy(imu))
        result = {}
        for branch in BRANCHES:
            gallery = dict(zip(data['accounts'], data['profiles'][branch]))
            accounts, distances = euclidean_scores(outputs[branch].numpy(), gallery)
            result[branch] = {'distances': distances.tolist(),
                              'thresholds': data['thresholds'][branch],
                              'accepted': {target: (distances <= threshold).tolist()
                                           for target, threshold in data['thresholds'][branch].items()}}
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, 'Inputs do not produce finite paired embeddings') from error
    return {'accounts': accounts, 'branches': result,
            'validation_status': data.get('validation_status'), 'missing': data.get('missing', []),
            'interpretation': 'joint encoder and same-checkpoint branch ablations'}
