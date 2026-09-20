"""Frozen paired neural BEACON research routes; mount only in research_api.app."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter(prefix='/api/neural-beacon', tags=['paired-neural-research'])


def artifact_dir(version):
    if version not in {'v1', 'v2'}:
        raise HTTPException(status_code=404, detail='Unknown frozen experiment')
    return ROOT / f'research/benchmarks/beacon-neural-{version}'


@lru_cache(maxsize=2)
def configuration(version):
    path = artifact_dir(version) / 'frozen.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Experiment has not frozen its models')
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def pairs(version):
    path = artifact_dir(version) / 'dev_pairs.npz'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Development validation has not completed')
    with np.load(path, allow_pickle=False) as a:
        return a['X'], a['y'], a['groups']


@lru_cache(maxsize=12)
def fitted(version, name):
    config = configuration(version)['models'].get(name)
    if config is None:
        raise HTTPException(status_code=404, detail='Unknown frozen model')
    path = artifact_dir(version) / f'{name}.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['model_sha256']:
        raise HTTPException(status_code=503, detail='Frozen model integrity check failed')
    return joblib.load(path)


@router.get('/{version}/info')
async def info(version: str):
    config = configuration(version)
    return {'version': version, 'models': config['models'], 'protocol': config['protocol'],
            'feature_width': max(max(c['columns']) for c in config['models'].values()) + 1,
            'score_direction': 'larger_is_impostor', 'test': 'sealed'}


@router.get('/{version}/pairs')
async def pair_page(version: str, split: str = 'dev', offset: int = Query(0, ge=0),
              limit: int = Query(128, ge=1, le=256)):
    if split != 'dev':
        raise HTTPException(status_code=403, detail='This frozen replay endpoint exposes development pairs only; test is sealed')
    x, y, groups = pairs(version)
    return {'version': version, 'split': 'dev', 'total': len(y), 'offset': offset,
            'x': x[offset:offset + limit].tolist(), 'y': y[offset:offset + limit].tolist(),
            'groups': groups[offset:offset + limit].tolist()}


class ScoreRequest(BaseModel):
    model: str = Field(min_length=1, max_length=64)
    features: list[list[float]] = Field(min_length=1, max_length=256)


@router.post('/{version}/score')
async def score(version: str, body: ScoreRequest):
    frozen = configuration(version)
    config = frozen['models'].get(body.model)
    if config is None:
        raise HTTPException(status_code=404, detail='Unknown frozen model')
    width = max(max(c['columns']) for c in frozen['models'].values()) + 1
    if any(len(row) != width for row in body.features):
        raise HTTPException(status_code=422, detail=f'Expected {width} paired features')
    x = np.asarray(body.features, dtype=float)
    if not np.isfinite(x).all() or (np.abs(x) > np.finfo(np.float32).max).any():
        raise HTTPException(status_code=422, detail='Paired features must be finite and float32-representable')
    values = -fitted(version, body.model).decision_function(x[:, config['columns']])
    thresholds = config['thresholds']
    return {'version': version, 'model': body.model, 'scores': values.tolist(),
            'score_direction': 'larger_is_impostor', 'thresholds': thresholds,
            'accepted': {name: (values <= threshold).tolist() for name, threshold in thresholds.items()},
            'threshold_source': 'frozen training calibration identities'}


@router.get('/{version}/results')
async def results(version: str):
    path = artifact_dir(version) / 'dev_results.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Development validation has not completed')
    return json.loads(path.read_text())
