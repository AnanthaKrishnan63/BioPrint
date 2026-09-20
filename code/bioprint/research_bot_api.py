"""Frozen bot-rule CMU replay; mount only behind research_app loopback guards."""
from functools import lru_cache
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from contracts import Sample
from engine.features import keystroke_vector, password_keystrokes
from eval.strict_cmu import load_partition
from bot_rule_validation import prepare, evaluate, digest

OUT = ROOT / 'research/benchmarks/bot_rules'
router = APIRouter(prefix='/api/bot-rules', tags=['frozen-bot-rule-replay'])


@lru_cache(maxsize=1)
def plan():
    specification = json.loads((OUT / 'frozen_plan.json').read_text())
    for name, path in [('bot', 'code/bioprint/engine/bot.py'),
                       ('features', 'code/bioprint/engine/features.py'),
                       ('reconstruction', 'scripts/production_api_replay.py')]:
        current = digest(ROOT / path)
        if current != specification['code_sha256'][name]:
            compatibility_path = OUT / 'api_source_compatibility.json'
            compatibility = json.loads(compatibility_path.read_text()) if compatibility_path.exists() else {}
            if not (name == 'bot' and compatibility.get('frozen_bot_sha256') == specification['code_sha256'][name]
                    and compatibility.get('compatible_bot_sha256') == current):
                raise HTTPException(503, 'Frozen validation implementation changed')
    return specification


@lru_cache(maxsize=1)
def priors():
    plan()
    names, train = load_partition('train')
    result = {}
    for owner, source in train.items():
        indices = np.flatnonzero(source['session'] == 1)
        indices = indices[np.argsort(source['rep'][indices])][:10]
        if len(indices) != 10:
            raise HTTPException(503, 'Incomplete frozen training enrollment')
        result[owner] = [prepare(names, row)[1].values for row in source['X'][indices]]
    return result


@lru_cache(maxsize=1)
def development():
    plan()
    names, source = load_partition('dev')
    rows = []
    for owner, records in source.items():
        for session, repetition, values in zip(records['session'], records['rep'], records['X']):
            sample, _ = prepare(names, values)
            rows.append({'claimed_owner': owner, 'session': int(session), 'repetition': int(repetition),
                         'sample': sample.model_dump(mode='json')})
    return rows


@router.get('/samples')
async def samples(split: str = 'dev', offset: int = Query(0, ge=0), limit: int = Query(128, ge=1, le=128)):
    if split != 'dev':
        raise HTTPException(403, 'Only genuine development reconstructions exposed; test is sealed')
    rows = development()
    return {'split': 'dev', 'total': len(rows), 'offset': offset, 'rows': rows[offset:offset + limit],
            'trust_assumed': True, 'environment_observed': False, 'pointer_observed': False}


class ClaimedSample(BaseModel):
    claimed_owner: str = Field(min_length=1, max_length=64)
    sample: Sample


class CheckRequest(BaseModel):
    rows: list[ClaimedSample] = Field(min_length=1, max_length=128)


@router.post('/infer')
async def infer(body: CheckRequest):
    history = priors()
    results = []
    for row in body.rows:
        if row.claimed_owner not in history:
            raise HTTPException(404, 'Unknown frozen CMU account')
        record = row.sample
        if len(record.keystrokes) > 256 or record.pointer or record.env:
            raise HTTPException(422, 'This replay contract requires bounded keyboard events and empty pointer/environment')
        if any(not np.isfinite(event.t) or abs(event.t) > 1e12 for event in record.keystrokes):
            raise HTTPException(422, 'Invalid keyboard event time')
        vector = keystroke_vector(password_keystrokes(record))
        results.append(evaluate(record, vector, history[row.claimed_owner]))
    return {'rows': results, 'history_source': 'first10 session1 repetitions only; never updated from probes',
            'scope': 'timing-rule replay; assumed event trust and empty environment do not validate browser trust'}


@router.get('/results')
async def results():
    specification = plan()
    result = json.loads((OUT / 'results.json').read_text())
    return {'plan': specification, 'genuine_dev': result['genuine_dev'], 'limitations': result['limitations'],
            'interpretation': 'Genuine false flags only; no bot-positive cohort, bot FAR/EER or browser trust validation'}
