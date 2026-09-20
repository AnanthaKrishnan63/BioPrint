"""Comparison-only SapiMouse Z-norm router; never a production login policy."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import research_pointer_api as pointer

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'research/benchmarks/pointer_sapimouse_znorm_dev_v2/release.json'
RELEASE_SHA256 = 'cb60dd7660b3ec201e123022ed00fbd5a7678806ee13b3b809a2c3166c409d1c'
router = APIRouter(prefix='/api/pointer-znorm', tags=['pointer normalization research'])


@lru_cache(maxsize=1)
def artifacts():
    if not RELEASE.exists() or RELEASE_SHA256 == 'UNRELEASED':
        raise HTTPException(503, 'Frozen comparison release unavailable')
    raw = RELEASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RELEASE_SHA256:
        raise HTTPException(503, 'Comparison manifest checksum mismatch')
    manifest = json.loads(raw)
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise HTTPException(503, 'Comparison artifact checksum mismatch')
    with np.load(RELEASE.parent / 'normalization.npz', allow_pickle=False) as data:
        accounts, mean, spread = data['accounts'].astype(str).tolist(), data['mean'], data['spread']
    if (len(accounts) != 24 or accounts != sorted(set(accounts)) or mean.shape != (24,)
            or spread.shape != (24,) or not np.isfinite(mean).all()
            or not np.isfinite(spread).all() or np.any(spread <= 0)):
        raise HTTPException(503, 'Invalid frozen normalization')
    return manifest, accounts, mean, spread


@router.get('/{split}/samples')
async def samples(split: str, offset: int = Query(0, ge=0), limit: int = Query(128, ge=1, le=256)):
    if split != 'dev':
        raise HTTPException(403, 'Only DEV observations are available; test is sealed')
    artifacts()
    return await pointer.sample_page('sapimouse', 'dev', offset, limit)


class Request(BaseModel):
    method: Literal['cosine', 'znorm'] = 'znorm'
    sequences: list[list[list[float]]] = Field(min_length=1, max_length=2048)
    session_ids: list[str] = Field(min_length=1, max_length=2048)


@router.post('/score')
async def score(body: Request):
    manifest, accounts, mean, spread = artifacts()
    original = await pointer.score('sapimouse', pointer.ScoreRequest(
        model='selected', sequences=body.sequences, session_ids=body.session_ids))
    if original['subjects'] != accounts or original['method'] != 'cosine@5':
        raise HTTPException(503, 'Original frozen cosine configuration changed')
    raw = np.asarray(original['scores'], dtype=np.float64).reshape(-1, len(accounts))
    values = (raw - mean) / spread if body.method == 'znorm' else raw
    if not np.isfinite(values).all(): raise HTTPException(422, 'Nonfinite normalized score')
    method = 'znorm_cosine@5' if body.method == 'znorm' else 'cosine@5'
    thresholds = manifest['thresholds'][method]
    return {**original, 'method': method, 'model': 'comparison_only',
            'scores': values.tolist(), 'thresholds': [thresholds['0.01']] * len(accounts),
            'accepted': (values >= thresholds['0.01']).tolist(),
            'accepted_at_targets': {t: (values >= v).tolist() for t, v in thresholds.items()},
            'training_selected_method': 'cosine@5', 'candidate_promoted': False,
            'evaluation_only': True}


@router.get('/results')
async def results():
    manifest, _, _, _ = artifacts()
    return manifest['comparison']
