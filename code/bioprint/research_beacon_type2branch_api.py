"""Frozen paired fusion replay; encoders run offline, not in these routes."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'research/benchmarks/beacon_type2branch_fusion_train_v1'
DEV = ROOT / 'research/benchmarks/beacon_type2branch_dev_evaluation_v1'
FROZEN_SHA256 = 'fb612db18d0d2635506df53d470b06cc9b5e4115efdc8731293a44a37367a1fe'
REPORT_SHA256 = '659f43709022ec41b4ac96cc76e4ee25751605749285e58c5204e21b64deb813'
router = APIRouter(prefix='/api/type2branch-beacon', tags=['paired-type2branch-research'])
BOUNDARY = 'Fusion replay on paired features from offline frozen encoders; not live browser capture or online encoder inference'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts():
    if REPORT_SHA256 is None:
        raise HTTPException(status_code=503, detail='Completed DEV evaluation is not pinned')
    try:
        if digest(TRAIN/'frozen.json') != FROZEN_SHA256 or digest(DEV/'report.json') != REPORT_SHA256:
            raise ValueError('Frozen artifact checksum mismatch')
        config = json.loads((TRAIN/'frozen.json').read_text())
        report = json.loads((DEV/'report.json').read_text())
        if (config['status'] != 'paired_train_fusion_complete' or len(config['models']) != 8
                or set(config['models']) != set(config['protocol']['models'])
                or report['status'] != 'paired_dev_evaluation_complete'
                or report['evaluation_split'] != 'dev' or report['frozen_sha256'] != FROZEN_SHA256
                or set(report['results']) != set(config['models'])):
            raise ValueError('Incomplete calibrated evaluation')
        for name, model in config['models'].items():
            columns = model['columns']
            if (not columns or columns != config['protocol']['models'][name]
                    or len(set(columns)) != len(columns)
                    or any(type(c) is not int or not 0 <= c < 35 for c in columns)
                    or set(model['thresholds']) != {'far_1pct', 'far_5pct', 'eer'}
                    or not all(np.isfinite(t) for t in model['thresholds'].values())
                    or digest(TRAIN/f'{name}.joblib') != model['model_sha256']):
                raise ValueError('Frozen model configuration mismatch')
        if digest(DEV/'dev_pairs.npz') != report['pairs_sha256']:
            raise ValueError('DEV pair checksum mismatch')
        return config, report
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=503, detail='Frozen research artifact verification failed') from error


def load_pairs(report):
    try:
        with np.load(DEV/'dev_pairs.npz', allow_pickle=False) as data:
            x, y, groups = data['X'], data['y'], data['groups']
        if (x.shape != (report['claims'], 35) or y.shape != (len(x),)
                or groups.shape != (len(x), 3) or not np.isfinite(x).all()
                or set(y.tolist()) != {0, 1}
                or not np.array_equal(y, groups[:, 0] == groups[:, 1])
                or set(groups[:, 0]) != set(report['subjects'])
                or set(groups[:, 1]) != set(report['subjects'])):
            raise ValueError('DEV pair contract mismatch')
        return x, y, groups
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=503, detail='Frozen DEV pairs are invalid') from error


@router.get('/info')
async def info():
    config, _ = artifacts()
    return {'models': config['models'], 'protocol': config['protocol'], 'feature_width': 35,
            'score_direction': 'larger_is_impostor', 'test': 'sealed', 'boundary': BOUNDARY}


@router.get('/pairs')
async def pair_page(split: str = 'dev', offset: int = Query(0, ge=0), limit: int = Query(128, ge=1, le=256)):
    if split != 'dev':
        raise HTTPException(status_code=403, detail='Only DEV pairs are exposed; test is sealed')
    _, report = artifacts()
    x, y, groups = load_pairs(report)
    return {'split': 'dev', 'total': len(y), 'offset': offset,
            'x': x[offset:offset+limit].tolist(), 'y': y[offset:offset+limit].tolist(),
            'groups': groups[offset:offset+limit].tolist()}


class ScoreRequest(BaseModel):
    model: str = Field(min_length=1, max_length=64)
    features: list[list[float]] = Field(min_length=1, max_length=256)


@router.post('/score')
async def score(body: ScoreRequest):
    if any(len(row) != 35 for row in body.features):
        raise HTTPException(status_code=422, detail='Expected35 paired features')
    x = np.asarray(body.features, dtype=np.float64)
    if not np.isfinite(x).all() or (np.abs(x) > np.finfo(np.float32).max).any():
        raise HTTPException(status_code=422, detail='Finite float32-representable features required')
    config, _ = artifacts()
    selected = config['models'].get(body.model)
    if selected is None:
        raise HTTPException(status_code=404, detail='Unknown frozen model')
    model = joblib.load(TRAIN/f'{body.model}.joblib')
    values = -np.asarray(model.decision_function(x[:, selected['columns']]), dtype=np.float64)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise HTTPException(status_code=503, detail='Frozen model produced invalid scores')
    return {'model': body.model, 'scores': values.tolist(), 'score_direction': 'larger_is_impostor',
            'thresholds': selected['thresholds'],
            'accepted': {name: (values <= threshold).tolist() for name, threshold in selected['thresholds'].items()},
            'threshold_source': 'frozen TRAIN calibration identities', 'boundary': BOUNDARY}


@router.get('/results')
async def results():
    _, report = artifacts()
    return {**report, 'boundary': BOUNDARY}
