"""Public-dataset pointer inference router for the isolated research API.

No production server/database imports; the parent research app supplies loopback
middleware. Artifact paths are fixed, never supplied by HTTP callers.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'scripts')]
router = APIRouter(prefix='/api/pointer', tags=['public pointer research'])
BAL = ROOT / 'research/benchmarks/pointer'
SAPI = ROOT / 'research/benchmarks/pointer_sapimouse'
OPT = ROOT / 'research/benchmarks/pointer_sapimouse_optimized'


def check_split(split):
    if split not in {'train', 'dev'}:
        raise HTTPException(403, 'Test contents are sealed')


@lru_cache(maxsize=2)
def spec(dataset):
    if dataset == 'balabit':
        source = BAL
        if not (source / 'results.json').exists():
            raise HTTPException(503, 'Balabit frozen validation is unavailable')
        frozen = json.loads((source / 'frozen_training_selection.json').read_text())
        users = sorted(p.stem.removeprefix('existing_scaled_manhattan_') for p in source.glob('existing_scaled_manhattan_user*.npz'))
        return {'source': source, 'frozen': frozen, 'subjects': users, 'methods': {'baseline': 'existing_scaled_manhattan', 'selected': frozen['selected']}, 'blocks': 1}
    if dataset == 'sapimouse':
        source = OPT if (OPT / 'results.json').exists() else SAPI
        if not (source / 'results.json').exists() or not (SAPI / 'encoder.torchscript.pt').exists():
            raise HTTPException(503, 'SapiMouse frozen training/export is not complete')
        frozen = json.loads((source / 'frozen_training_selection.json').read_text())
        selected = frozen['selected']
        blocks = int(selected.split('@')[1])
        manifest = json.loads((ROOT / 'data/benchmarks/sapimouse/split_manifest.json').read_text())
        users = sorted({e['path'].split('/')[1] for e in manifest['files'] if e['role'] == 'dev_support_enrollment' and e['split'] == 'train'})
        return {'source': source, 'frozen': frozen, 'subjects': users, 'methods': {'baseline': f'scaled_manhattan@{blocks}', 'selected': selected, 'author': f'ocsvm:0.5:raw@{blocks}'}, 'blocks': blocks}
    raise HTTPException(404, 'Unknown pointer dataset')


@lru_cache(maxsize=4)
def samples(dataset, split):
    check_split(split)
    spec(dataset)
    if dataset == 'balabit':
        from pointer_benchmark import load_entries, DATA
        manifest = json.loads((DATA / 'split_manifest.json').read_text())
        x, labels, groups = load_entries([e for e in manifest['entries'] if e['split'] == split])
        return x, None, labels, groups
    from pointer_sapimouse_benchmark import load, DATA
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    sequence, hand, labels, groups = load([e for e in manifest['files'] if e['split'] == split])
    return hand, sequence, labels, groups


@lru_cache(maxsize=1)
def encoder():
    import torch
    from engine.pointer_sequence import PointerSequenceEncoder
    torch.set_num_threads(2)
    checkpoint = SAPI / 'encoder.torchscript.pt'
    manifest = json.loads((SAPI / 'encoder_manifest.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['sha256']:
        raise HTTPException(503, 'Pointer encoder artifact hash mismatch')
    return PointerSequenceEncoder(checkpoint)


@lru_cache(maxsize=128)
def model_artifact(dataset, method, user):
    import joblib
    if dataset == 'balabit':
        if method == 'existing_scaled_manhattan':
            with np.load(BAL / f'{method}_{user}.npz', allow_pickle=False) as a:
                return {'center': a['center'], 'spread': a['spread'], 'cap': float(a['cap'])}
        estimator = joblib.load(BAL / f'{method}_{user}.joblib')
        # Runtime-only change: all dev scores/decisions verified before enabling.
        estimator.set_params(n_jobs=1)
        return estimator
    if method in {'scaled_manhattan', 'cosine', 'latent_manhattan'}:
        path = (SAPI if method == 'scaled_manhattan' else OPT) / f'{method}_{user}.npz'
        with np.load(path, allow_pickle=False) as a:
            return {key: a[key] for key in a.files}
    return joblib.load(SAPI / f'{method.replace(":", "_")}_{user}.joblib')


@router.get('/datasets')
async def datasets():
    names = ['balabit'] if (BAL / 'results.json').exists() else []
    if (SAPI / 'encoder.torchscript.pt').exists() and (SAPI / 'results.json').exists():
        names.append('sapimouse')
    return {'datasets': names, 'splits': ['train', 'dev'], 'test': 'sealed'}


@router.get('/{dataset}/{split}/samples')
async def sample_page(dataset: str, split: str, offset: int = Query(0, ge=0), limit: int = Query(128, ge=1, le=256)):
    check_split(split)
    x, sequence, labels, groups = samples(dataset, split)
    end = min(offset + limit, len(x))
    feature_rows = x[offset:end].astype(object)
    feature_rows[~np.isfinite(x[offset:end])] = None
    return {'dataset': dataset, 'split': split, 'total': len(x), 'offset': offset, 'features': feature_rows.tolist(), 'sequences': None if sequence is None else sequence[offset:end].tolist(), 'labels': labels[offset:end].tolist(), 'session_ids': groups[offset:end].tolist(), 'subjects': spec(dataset)['subjects']}


class ScoreRequest(BaseModel):
    model: Literal['baseline', 'selected', 'author'] = 'selected'
    features: list[list[float | None]] | None = Field(default=None, max_length=2048)
    sequences: list[list[list[float]]] | None = Field(default=None, max_length=2048)
    session_ids: list[str] = Field(min_length=1, max_length=2048)


def input_array(value, dtype=float):
    try:
        return np.asarray(value, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, 'Input arrays must be rectangular numeric values') from exc


@router.post('/{dataset}/score')
async def score(dataset: str, body: ScoreRequest):
    cfg = spec(dataset)
    if body.model not in cfg['methods']:
        raise HTTPException(404, 'Model is unavailable for this dataset')
    name = cfg['methods'][body.model]
    method = name.split('@')[0]
    subjects = cfg['subjects']
    if dataset == 'balabit' or method == 'scaled_manhattan':
        x = input_array(body.features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 29 or not len(x) or np.isinf(x).any():
            raise HTTPException(422, 'Expected finite/missing 29-feature pointer vectors')
        if np.any(np.abs(x[np.isfinite(x)]) > np.finfo(np.float32).max):
            raise HTTPException(422, 'Pointer features exceed supported float32 range')
        if dataset == 'sapimouse':
            medians = np.load(SAPI / 'training_feature_medians.npy', allow_pickle=False)
            x = np.where(np.isnan(x), medians, x)
        elif not np.isfinite(x).all():
            raise HTTPException(422, 'Balabit features must all be finite')
    else:
        a = input_array(body.sequences, dtype=np.float32)
        if a.ndim != 3 or a.shape[1:] != (2, 128) or not len(a) or not np.isfinite(a).all():
            raise HTTPException(422, 'Expected finite normalized sequence blocks with shape (n,2,128)')
        x = encoder().encode_blocks(a)
    if len(x) != len(body.session_ids):
        raise HTTPException(422, 'Each input block must name its session')
    columns = []
    for user in subjects:
        artifact = model_artifact(dataset, method, user)
        if method in {'existing_scaled_manhattan', 'scaled_manhattan', 'latent_manhattan'}:
            deviations = np.abs(x - artifact['center']) / artifact['spread']
            values = -np.minimum(deviations, float(artifact['cap'])).mean(axis=1)
        elif method == 'cosine':
            values = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)) @ artifact['center']
        elif dataset == 'balabit':
            values = artifact.predict_proba(x)[:, 1]
        else:
            values = artifact.score_samples(x)
            if method.endswith(':enroll'):
                values = values / (artifact.nu * artifact.shape_fit_[0])
        columns.append(values)
    matrix = np.column_stack(columns)
    groups = np.asarray(body.session_ids)
    aggregated, starts, output_groups = [], [], []
    for group in dict.fromkeys(body.session_ids):
        indices = np.flatnonzero(groups == group)
        for start in range(0, len(indices) - cfg['blocks'] + 1, cfg['blocks']):
            block = indices[start:start + cfg['blocks']]
            aggregated.append(matrix[block].mean(axis=0))
            starts.append(int(block[0]))
            output_groups.append(group)
    scores = np.asarray(aggregated).reshape(-1, len(subjects))
    threshold = cfg['frozen']['thresholds'][name]['0.01']
    return {'dataset': dataset, 'model': body.model, 'method': name, 'subjects': subjects, 'scores': scores.tolist(), 'thresholds': [threshold] * len(subjects), 'accepted': (scores >= threshold).tolist(), 'input_start_indices': starts, 'session_ids': output_groups, 'blocks_per_decision': cfg['blocks'], 'coordinates_per_contiguous_decision': 128 * cfg['blocks'] + 1 if dataset == 'sapimouse' else None, 'score_direction': 'larger_is_genuine', 'threshold_source': 'frozen training calibration at1% empirical FAR target; achieved dev FAR may differ'}


@router.get('/{dataset}/results')
async def results(dataset: str):
    cfg = spec(dataset)
    return json.loads((cfg['source'] / 'results.json').read_text())


@router.get('/dashboard', response_class=HTMLResponse)
async def dashboard():
    return '''<!doctype html><meta charset="utf-8"><title>Pointer validation</title><style>body{font:16px system-ui;max-width:960px;margin:3rem auto}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Pointer development validation</h1><p>Models and thresholds are selected on training data. Sealed test records are not accessed.</p><table><thead><tr><th>Dataset / model</th><th>Pooled EER</th><th>FRR</th><th>FAR</th></tr></thead><tbody id="rows"></tbody></table><p>Mouse sequences require many events; these are not single-click guarantees. Dataset device/task differences may aid recognition.</p><p>Macro per-user EER differs: author OCSVM 7.73%, selected cosine 8.27%. The table reports pooled EER and frozen global operating points.</p><script>(async()=>{let d=await(await fetch('/api/pointer/datasets')).json();for(let name of d.datasets){let r=await(await fetch('/api/pointer/'+name+'/results')).json();for(let [model,points] of Object.entries(r.dev)){let m=points['0.01'];let row=document.createElement('tr');for(let value of [name+' / '+model,...['eer_descriptive','frr','far'].map(k=>(m[k]*100).toFixed(2)+'%')]){let c=document.createElement('td');c.textContent=value;row.append(c)}document.querySelector('#rows').append(row)}}})()</script>'''
