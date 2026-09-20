"""Isolated loopback-only public-dataset replay service. Never imports server/db.

Opt in explicitly:
  PYTHONPATH=.research-deps:code/bioprint ~/miniconda3/envs/bigidea/bin/python \
    -m uvicorn research_api:app --host 127.0.0.1 --port 8011

This service intentionally cannot be reached through the participant LAN app.
"""
from __future__ import annotations

from functools import lru_cache
from contextlib import asynccontextmanager
import ipaddress
import json
from pathlib import Path
import sys
from typing import Literal
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'scripts')]

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from data_keystrokes import checksum, iter_rows
from keystroke_benchmark import load_track, flatten

ARTIFACTS = ROOT / 'research/benchmarks/results/keyrecs-v1'
DATASETS = {'keyrecs-fixed': 'fixed', 'keyrecs-free': 'free'}
CMU_ARTIFACTS = ROOT / 'research/benchmarks/cmu'
CMU_DATASETS = {'cmu': CMU_ARTIFACTS,
                'cmu-account-selection': ROOT / 'research/benchmarks/cmu-account-selection',
                'cmu-far-selection': ROOT / 'research/benchmarks/cmu-far-selection'}
TYPENET_ARTIFACTS = ROOT / 'research/benchmarks/results/typenet-keyrecs-v1'


@asynccontextmanager
async def research_lifespan(app):
    try:
        yield
    finally:
        try:
            close_type2branch_worker()
        finally:
            close_type2branch_capture_worker()


app = FastAPI(title='BioPrint research replay', docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=research_lifespan)
from research_modalities import router as paired_router, tabular_router
app.include_router(paired_router)
app.include_router(tabular_router)
from research_pointer_api import router as pointer_router
app.include_router(pointer_router)
from research_beacon_neural_api import router as neural_paired_router
app.include_router(neural_paired_router)
from research_beacon_type2branch_api import router as type2branch_paired_router
app.include_router(type2branch_paired_router)
from research_bot_api import router as bot_rules_router
app.include_router(bot_rules_router)
from research_hmog_api import router as hmog_router
app.include_router(hmog_router)
from research_pointer_znorm_api import router as pointer_znorm_router
app.include_router(pointer_znorm_router)
from research_arithmetic_api import router as arithmetic_router
app.include_router(arithmetic_router)
from research_type2branch_api import router as type2branch_router, close_worker as close_type2branch_worker
app.include_router(type2branch_router)
from research_type2branch_capture_api import router as type2branch_capture_router, close_worker as close_type2branch_capture_worker
app.include_router(type2branch_capture_router)


def _local_host(url: str) -> bool:
    try:
        hostname = urlsplit(url).hostname
        return hostname in {'localhost', '127.0.0.1', '::1'}
    except ValueError:
        return False


@app.middleware('http')
async def loopback_only(request: Request, call_next):
    try:
        is_loopback = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        is_loopback = False
    origin = request.headers.get('origin')
    if not is_loopback or not _local_host(str(request.url)) or (origin and not _local_host(origin)):
        return JSONResponse({'detail': 'Research API is restricted to localhost'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'"
    return response


def check_split(split: str) -> None:
    if split not in {'train', 'dev'}:
        raise HTTPException(status_code=403, detail='Test data is sealed; only train/dev reads are allowed')


def track_for(dataset: str) -> str:
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail='Unknown dataset')
    return DATASETS[dataset]


@lru_cache(maxsize=3)
def cmu_artifact(dataset='cmu'):
    path = CMU_DATASETS[dataset] / 'frozen_models.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='CMU model has not frozen')
    return json.loads(path.read_text())


def cmu_model_labels(dataset, artifact):
    labels = {'baseline': 'baseline', 'train_selected': artifact['selected']}
    if dataset != 'cmu':
        labels['global_reference'] = artifact['protocol']['original_selected']
    if dataset == 'cmu-far-selection':
        labels['eer_reference'] = 'account_selected'
    return labels


@lru_cache(maxsize=8)
def frozen(dataset: str):
    if dataset == 'keyrecs-typenet':
        path = TYPENET_ARTIFACTS / 'frozen.json'
        if not path.exists():
            raise HTTPException(status_code=503, detail='TypeNet training has not frozen a model')
        spec = json.loads(path.read_text())
        with np.load(TYPENET_ARTIFACTS / 'frozen-arrays.npz', allow_pickle=False) as values:
            thresholds = values['thresholds'].tolist()
        return {'subjects': spec['subjects'], 'feature_names': [f'event{i}.{f}' for i in range(50)
                for f in ['keycode', 'hold', 'interkey', 'press', 'release']],
                'frozen': {'train_selected': {'thresholds': thresholds}}}
    if dataset in CMU_DATASETS:
        artifact = cmu_artifact(dataset)
        subjects = sorted(artifact['models'][artifact['selected']])
        labels = cmu_model_labels(dataset, artifact)
        return {'subjects': subjects,
                'feature_names': artifact['models'][artifact['selected']][subjects[0]]['names'],
                'frozen': {label: {'thresholds': [-artifact['models'][method][s]['calibration']['far_1pct']
                                                for s in subjects]} for label, method in labels.items()}}
    track = track_for(dataset)
    path = ARTIFACTS / f'{track}-frozen.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Frozen model has not been built')
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def samples(dataset: str, split: str):
    check_split(split)
    if dataset == 'keyrecs-typenet':
        from typenet_benchmark import sequences
        config = frozen(dataset)
        data, audit = sequences(split)
        x, y = flatten(data, config['subjects'])
        return x.reshape(len(x), 250), y
    if dataset in CMU_DATASETS:
        from eval.strict_cmu import load_partition
        names, records = load_partition(split)
        config = frozen(dataset)
        if names != config['feature_names']:
            raise HTTPException(status_code=503, detail='CMU feature contract mismatch')
        return flatten({s: records[s]['X'] for s in config['subjects']}, config['subjects'])
    track = track_for(dataset)
    config = frozen(dataset)
    if split == 'dev':
        with np.load(ARTIFACTS / f'{track}-dev-features.npz', allow_pickle=False) as archive:
            x, y = archive['x'], archive['y']
            if archive['subjects'].tolist() != config['subjects']:
                raise HTTPException(status_code=503, detail='Artifact subject order mismatch')
    else:
        data, names = load_track(track, 'train', config['free_window'])
        if names != config['feature_names']:
            raise HTTPException(status_code=503, detail='Artifact feature contract mismatch')
        x, y = flatten(data, config['subjects'])
    return x, y


@lru_cache(maxsize=8)
def fitted_model(dataset: str, model: str):
    import joblib
    if dataset == 'keyrecs-typenet':
        if model != 'train_selected':
            raise HTTPException(status_code=404, detail='TypeNet supports only its frozen train-selected model')
        return TypeNetVerifier()
    if dataset in CMU_DATASETS:
        if model not in frozen(dataset)['frozen']:
            raise HTTPException(status_code=404, detail='Unknown CMU model')
        artifact = cmu_artifact(dataset)
        method = cmu_model_labels(dataset, artifact)[model]
        return CMUVerifier(artifact['models'][method], frozen(dataset)['subjects'])
    track = track_for(dataset)
    if model not in {'reference_logistic', 'train_selected'}:
        raise HTTPException(status_code=404, detail='Unknown frozen model')
    # Only operator-produced files at fixed local paths can be deserialized.
    # No upload or request-supplied path is accepted.
    fitted = joblib.load(ARTIFACTS / f'{track}-{model}.joblib')
    if getattr(fitted, 'kind', None) == 'extra_trees':
        # Scheduling only: runtime-audit.json verifies all 981,259 dev scores
        # and every frozen decision, including the closest threshold tie.
        # Avoid worker startup overhead for small interactive requests.
        fitted.model.n_jobs = 1
    return fitted


class CMUVerifier:
    def __init__(self, models, subjects):
        self.models, self.subjects = models, subjects

    def scores(self, x):
        from eval.strict_cmu import distances
        # Keep the API's score direction uniform across datasets.
        return -np.column_stack([distances(self.models[s], x) for s in self.subjects])


class TypeNetVerifier:
    def __init__(self):
        import torch
        from typenet_benchmark import TypeNet, TimingTransform
        torch.set_num_threads(2)
        spec = json.loads((TYPENET_ARTIFACTS / 'frozen.json').read_text())
        if checksum(TYPENET_ARTIFACTS / 'model.pt') != spec['model_sha256']:
            raise HTTPException(status_code=503, detail='TypeNet checkpoint hash mismatch')
        self.model = TypeNet()
        self.model.load_state_dict(torch.load(TYPENET_ARTIFACTS / 'model.pt', weights_only=True))
        self.transform = TimingTransform()
        with np.load(TYPENET_ARTIFACTS / 'frozen-arrays.npz', allow_pickle=False) as a:
            self.transform.median, self.transform.lo, self.transform.hi = a['median'], a['lo'], a['hi']
            self.gallery = a['gallery']

    def scores(self, x):
        from typenet_benchmark import embeddings, gallery_scores
        a = self.transform.apply(x.reshape(-1, 50, 5).astype(np.float32))
        return gallery_scores(embeddings(self.model, a), self.gallery)


@app.get('/api/datasets')
def dataset_list():
    available = [*DATASETS, 'cmu']
    for dataset, directory in CMU_DATASETS.items():
        if dataset != 'cmu' and (directory / 'dev_results.json').exists():
            available.append(dataset)
    if (TYPENET_ARTIFACTS / 'dev-results.json').exists():
        available.append('keyrecs-typenet')
    return {'datasets': available, 'readable_splits': ['train', 'dev'],
            'test': 'sealed', 'purpose': 'public research datasets; never production users'}


@app.get('/api/datasets/{dataset}/{split}/samples')
def sample_page(dataset: str, split: str, offset: int = Query(0, ge=0),
                limit: int = Query(128, ge=1, le=256)):
    check_split(split)
    x, y = samples(dataset, split)
    config = frozen(dataset)
    end = min(offset + limit, len(x))
    # JSON nulls represent missing measurements; scoring imputes with fit-only medians.
    vals = x[offset:end].astype(object)
    vals[~np.isfinite(x[offset:end])] = None
    return {'dataset': dataset, 'split': split, 'offset': offset, 'total': len(x),
            'subjects': config['subjects'], 'feature_names': config['feature_names'],
            'x': vals.tolist(), 'y': y[offset:end].tolist()}


@app.get('/api/datasets/{dataset}/{split}/raw')
def raw_page(dataset: str, split: str, offset: int = Query(0, ge=0),
             limit: int = Query(128, ge=1, le=256)):
    check_split(split)
    if dataset == 'keyrecs-typenet':
        # Same public source timing rows; /samples supplies the mapped-keycode
        # sequence contract used by this architecture.
        dataset = 'keyrecs-free'
    if dataset in CMU_DATASETS:
        from eval.strict_cmu import load_partition
        names, records = load_partition(split)
        rows = []
        index = 0
        for subject, data in records.items():
            for session, repetition, values in zip(data['session'], data['rep'], data['X']):
                if offset <= index < offset + limit:
                    rows.append([subject, int(session), int(repetition), *values.tolist()])
                index += 1
        return {'dataset': dataset, 'split': split, 'offset': offset,
                'header': ['subject', 'session', 'repetition', *names], 'rows': rows,
                'note': 'Source timing projection: Enter omitted; seconds converted to milliseconds'}
    track = track_for(dataset)
    rows, header = [], []
    for index, (columns, values) in enumerate(iter_rows(track, split)):
        if index < offset:
            continue
        if len(rows) >= limit:
            break
        header = columns
        rows.append(values)
    return {'dataset': dataset, 'split': split, 'offset': offset, 'header': header, 'rows': rows,
            'note': 'Timing source rows; free-text key labels intentionally omitted; test skipped before parsing'}


class ScoreBody(BaseModel):
    features: list[list[float | None]] = Field(min_length=1, max_length=256)
    model: Literal['reference_logistic', 'baseline', 'train_selected', 'global_reference', 'eer_reference'] = 'train_selected'


@app.post('/api/models/{dataset}/score')
def score(dataset: str, body: ScoreBody):
    config = frozen(dataset)
    dim = len(config['feature_names'])
    if any(len(row) != dim for row in body.features):
        raise HTTPException(status_code=422, detail=f'Expected exactly {dim} features per sample')
    x = np.asarray(body.features, dtype=float)
    if np.isinf(x).any():
        raise HTTPException(status_code=422, detail='Infinite input is not supported')
    if dataset in CMU_DATASETS and not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail='CMU models require finite timing features')
    if dataset in CMU_DATASETS and (np.abs(x) > np.finfo(np.float32).max).any():
        raise HTTPException(status_code=422, detail='CMU timing features exceed the supported numeric range')
    if dataset == 'keyrecs-typenet':
        keycodes = x[:, ::5]
        if not np.isfinite(keycodes).all() or (keycodes < 0).any() or (keycodes > 1).any():
            raise HTTPException(status_code=422, detail='TypeNet normalized keycodes must be finite values in [0,1]')
    scores = fitted_model(dataset, body.model).scores(x)
    thresholds = np.asarray(config['frozen'][body.model]['thresholds'])
    return {'dataset': dataset, 'model': body.model, 'subjects': config['subjects'],
            'scores': scores.tolist(), 'thresholds': thresholds.tolist(),
            'accepted': (scores >= thresholds).tolist(), 'score_direction': 'larger_is_genuine',
            'threshold_source': 'frozen training calibration; never calibrated on dev'}


@app.get('/api/results/{dataset}')
def results(dataset: str):
    if dataset == 'keyrecs-typenet':
        path = TYPENET_ARTIFACTS / 'dev-results.json'
        if not path.exists():
            raise HTTPException(status_code=503, detail='TypeNet validation has not completed')
        raw = json.loads(path.read_text())
        return {**raw, 'validation': {'train_selected': raw['validation']}}
    if dataset in CMU_DATASETS:
        raw = json.loads((CMU_DATASETS[dataset] / 'dev_results.json').read_text())
        validation = {}
        for method, row in raw['results'].items():
            summary = row['summary']
            validation[method] = {'aggregate_macro_user': {
                'eer': summary['macro_eer_diagnostic'], **summary['operating_points']['far_1pct']}}
        return {'dataset': dataset, 'evaluation_split': 'dev', 'validation': validation,
                'protocol': raw['protocol'], 'note': 'Legacy exposed dataset; no pristine holdout claim'}
    track = track_for(dataset)
    path = ARTIFACTS / f'{track}-results.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Validation has not completed')
    result = json.loads(path.read_text())
    result['result_sha256'] = checksum(path)
    return result


@app.get('/', response_class=HTMLResponse)
def homepage():
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>BioPrint research validation</title><style>body{font:16px system-ui;max-width:950px;margin:3rem auto;padding:1rem;color:#172436}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #ccc}small{color:#526176}</style>
<h1>BioPrint research validation</h1><p>Public-dataset development results. Test identities remain sealed. The participant app and its database are separate.</p>
<p><a href="/api/features/dashboard">Device, bot, cognitive, and paired-modality validation</a></p>
<p><a href="/api/pointer/dashboard">Pointer validation</a></p>
<p><a href="/api/bot-rules/results">Bot-rule genuine false-flag audit and limitations</a></p>
<p>Paired neural transfer: <a href="/api/neural-beacon/v1/results">TypeNet + handcrafted mouse</a> · <a href="/api/neural-beacon/v2/results">TypeNet + SapiMouse FCN</a>. Frozen cross-dataset transfers did not improve the deployment operating point.</p>
<table><thead><tr><th>Track / model</th><th>Macro EER</th><th>FRR</th><th>Achieved FAR</th></tr></thead><tbody id="metrics"></tbody></table>
<p><small>FRR and FAR use thresholds frozen on training calibration. EER is a diagnostic sweep on development scores. These numbers are not a production security guarantee.</small></p>
<pre id="status">Loading frozen validation reports…</pre><script>
const pct=x=>(100*x).toFixed(2)+'%';
(async()=>{const d=await(await fetch('/api/datasets')).json();for(const name of d.datasets){const r=await(await fetch('/api/results/'+name)).json();for(const [model,v] of Object.entries(r.validation||{})){const a=v.aggregate_macro_user;const row=document.createElement('tr');for(const val of [name+' / '+model,pct(a.eer),pct(a.frr),pct(a.far)]){const c=document.createElement('td');c.textContent=val;row.append(c)}document.querySelector('#metrics').append(row)}}document.querySelector('#status').textContent='API: GET /api/datasets/{dataset}/dev/samples; POST /api/models/{dataset}/score';})().catch(e=>document.querySelector('#status').textContent=String(e));
</script></html>'''
