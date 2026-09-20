"""Frozen arithmetic research outcome; incomplete DEV cannot supply predictions."""
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/benchmarks'
PINS = {
    'arithmetic_train_v1/train_complete.json': '78b4dc4db01c2e93adf298f75ecdd5eac0ea9675dc941d64cb9373bc7f77d5e9',
    'arithmetic_dev_failure_v1/report.json': 'a065cefeaf21ae3c92195e07fe855c6a18ca262695fc082860bacb2e5d77b89b',
    'arithmetic_dev_v1/plan.json': 'c3152ebae1d047124a7c18884b24dfce0380c65d2164a0c15c960c4f80045014',
}
router = APIRouter(prefix='/api/arithmetic', tags=['arithmetic research'])


def artifacts():
    loaded = []
    for name, expected in PINS.items():
        try:
            raw = (BASE / name).read_bytes()
        except OSError as exc:
            raise HTTPException(503, 'Frozen arithmetic evidence unavailable') from exc
        if hashlib.sha256(raw).hexdigest() != expected:
            raise HTTPException(503, 'Frozen arithmetic evidence checksum mismatch')
        loaded.append(json.loads(raw))
    train, failure, plan = loaded
    if (failure['status'] != 'infeasible' or failure['metrics'] != {}
            or failure['subset_scored'] or failure['parser_changed']
            or failure['test_accessed'] or plan['selected'] != train['selected']
            or failure['selected'] != train['selected']):
        raise HTTPException(503, 'Arithmetic outcome no longer matches frozen release')
    return train, failure


@router.get('/results')
async def results():
    train, failure = artifacts()
    return {'dataset': 'arithmetic', 'status': 'infeasible',
            'training_selected_method': train['selected'],
            'training_selection': train['selection'],
            'training_calibration': train['calibration'],
            'validation': failure, 'validation_metrics_available': False,
            'scoring_available': False, 'test_accessed': False,
            'limitations': ['One visit with ordered difficulty blocks.',
                            'Missing required DEV enrollment prevents complete-cohort recognition.',
                            'TRAIN calibration rates are not DEV performance.',
                            'No all-feature fusion or established cognitive SOTA claim.']}


def unavailable():
    _, failure = artifacts()
    raise HTTPException(409, {'status': 'infeasible',
                             'reason': 'Required DEV enrollment violates frozen block contract',
                             'invalid_blocks': failure['invalid_blocks'],
                             'validation_metrics_available': False})


@router.get('/{split}/samples')
async def samples(split: str):
    if split != 'dev':
        raise HTTPException(403, 'Only DEV requested; test observations remain sealed')
    unavailable()


@router.post('/score')
async def score():
    unavailable()
