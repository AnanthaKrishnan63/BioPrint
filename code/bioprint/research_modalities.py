"""Frozen paired-modality inference, mounted only in the isolated research app."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
BEACON = ROOT / 'research/benchmarks/beacon'
router = APIRouter(prefix='/api/paired/beacon')
tabular_router = APIRouter(prefix='/api/features')
TABULAR = {'device', 'delbot', 'cognitive', 'touch_tsi', 'beacon_nonlinear'}
NONLINEAR_EXPORT_MANIFEST_SHA256 = '99514fc65662307bad9826e6dbf2c6030706ef5d7434579de5081e5f296f6ace'
SINGLETHREAD_MODELS = {
    ('device', 'fpstalker_inspired_random_forest'),
    ('delbot', 'geometric_random_forest'),
    ('cognitive', 'learned_full_profile_rf'),
    ('touch_tsi', 'selected'),
}


@tabular_router.get('/summary')
async def summary_reports():
    rows = []
    for dataset in sorted(TABULAR):
        path = ROOT / f'research/benchmarks/{dataset}_results.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for name, method in report['methods'].items():
            rates = method['operating_points']['0.01']['dev']
            rows.append({'dataset': dataset, 'model': name,
                         'eer': method['dev_eer_descriptive'],
                         'far': rates['far'], 'frr': rates['frr']})
    path = BEACON / 'dev_results.json'
    if path.exists():
        for name, method in json.loads(path.read_text())['results'].items():
            rates = method['operating_points']['far_1pct']
            rows.append({'dataset': 'beacon', 'model': name,
                         'eer': method['eer_diagnostic'], **rates})
    return {'rows': rows, 'threshold_source': 'training calibration',
            'comparison_warning': 'Different datasets and tasks; compare models within a dataset only.'}


@tabular_router.get('/dashboard', response_class=HTMLResponse)
async def feature_dashboard():
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>BioPrint modality validation</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}table{border-collapse:collapse;width:100%}</style>
<a href="/">Typing validation</a><h1>Modality validation</h1>
<p>Frozen development results. Compare methods within each dataset; these tasks and populations differ.</p>
<p>Device results measure browser linkage. Cognitive tasks are repeated Stroop/Flanker trials, not keypad login challenges. Touch TSI measures keyboard tap locations and intervals, without randomized-keypad reaction times. BEACON pairs gameplay keyboard, mouse, and context; it excludes keypad and bot evidence.</p>
<table><thead><tr><th>Dataset / model</th><th>Diagnostic EER</th><th>FRR</th><th>Actual FAR</th></tr></thead><tbody id="rows"></tbody></table>
<p>Operating thresholds target 1% FAR on training calibration. Actual dev FAR can differ sharply. EER does not set the deployed threshold. Negative results are retained.</p>
<p id="status">Loading frozen reports…</p><script>
(async()=>{const r=await fetch('/api/features/summary');if(!r.ok)throw Error('Report unavailable');const d=await r.json();for(const v of d.rows){const tr=document.createElement('tr');for(const text of [v.dataset+' / '+v.model,...[v.eer,v.frr,v.far].map(x=>(100*x).toFixed(2)+'%')]){const td=document.createElement('td');td.textContent=text;tr.append(td)}document.querySelector('#rows').append(tr)}document.querySelector('#status').textContent='Test recordings remain sealed. No live account data appear here.'})().catch(e=>document.querySelector('#status').textContent=String(e));
</script></html>'''


def artifact_directory(dataset):
    if dataset not in TABULAR:
        raise HTTPException(404, 'Unknown dataset')
    directory = ROOT / 'research/benchmarks' / dataset
    if not (directory / 'schema.json').exists():
        raise HTTPException(503, 'Frozen artifact is not ready')
    return directory


def verified_nonlinear_export():
    """Validate the pinned export on cache misses, before deserializing models.

    Cached model objects are not continuously rechecked. Loading from verified
    bytes avoids reopening a model file between its hash check and joblib load.
    """
    directory = ROOT / 'research/benchmarks/beacon_nonlinear'
    try:
        raw_manifest = (directory / 'export_integrity.json').read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != NONLINEAR_EXPORT_MANIFEST_SHA256:
            raise ValueError('Manifest hash mismatch')
        manifest = json.loads(raw_manifest)
        if manifest['version'] != 1 or manifest['dataset'] != 'beacon_nonlinear':
            raise ValueError('Unexpected export manifest')
        verified = {}
        for relative, expected in manifest['files'].items():
            path = ROOT / relative
            path.resolve().relative_to(ROOT.resolve())
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Export hash mismatch')
            verified[relative] = raw
        return verified
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, 'Nonlinear export integrity check failed') from None


@lru_cache(maxsize=5)
def tabular_schema(dataset):
    if dataset == 'beacon_nonlinear':
        verified = verified_nonlinear_export()
        return json.loads(verified['research/benchmarks/beacon_nonlinear/schema.json'])
    return json.loads((artifact_directory(dataset) / 'schema.json').read_text())


@lru_cache(maxsize=12)
def tabular_model(dataset, name):
    import joblib
    schema = tabular_schema(dataset)
    if name not in schema['models']:
        raise HTTPException(404, 'Unknown frozen model')
    filename = schema['models'][name]['artifact']
    if Path(filename).name != filename:
        raise HTTPException(503, 'Invalid artifact path')
    if dataset == 'beacon_nonlinear':
        verified = verified_nonlinear_export()
        key = 'research/benchmarks/beacon_nonlinear/' + filename
        if key not in verified:
            raise HTTPException(503, 'Nonlinear model is absent from integrity manifest')
        model = joblib.load(io.BytesIO(verified[key]))
    else:
        model = joblib.load(artifact_directory(dataset) / filename)
    if (dataset, name) in SINGLETHREAD_MODELS:
        # runtime_threads.json: generated-input latency ~4x lower; all dev
        # decisions and frozen FAR/FRR identical. No fitted weights change.
        model.set_params(n_jobs=1)
    return model


@lru_cache(maxsize=1)
def configuration():
    return json.loads((BEACON / 'frozen.json').read_text())


@lru_cache(maxsize=4)
def model_for(name):
    import joblib
    if name not in configuration()['models']:
        raise HTTPException(404, 'Unknown frozen model')
    return joblib.load(BEACON / f'{name}.joblib')


class PairBatch(BaseModel):
    features: list[list[float]] = Field(min_length=1, max_length=256)


def feature_matrix(batch, width):
    try:
        x = np.asarray(batch.features, dtype=float)
    except (ValueError, OverflowError):
        raise HTTPException(422, 'Inconsistent or unrepresentable feature values') from None
    if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
        raise HTTPException(422, 'Incorrect feature dimensions or nonfinite values')
    # sklearn tree inference converts to float32; reject rather than silently
    # clamp values that cannot survive that conversion. Also bounds reductions.
    if np.any(np.abs(x) > np.finfo(np.float32).max):
        raise HTTPException(422, 'Feature magnitude exceeds supported float32 range')
    return x


@tabular_router.get('/{dataset}/{split}/samples')
async def tabular_samples(dataset: str, split: str, offset: int = Query(0, ge=0),
                          limit: int = Query(128, ge=1, le=256)):
    if split not in {'train', 'dev'}:
        raise HTTPException(403, 'Test data is sealed')
    if split != 'dev':
        raise HTTPException(404, 'Only frozen dev feature export is available')
    with np.load(artifact_directory(dataset) / 'dev_features.npz', allow_pickle=False) as data:
        return {'features': data['X'][offset:offset+limit].tolist(),
                'genuine': data['y'][offset:offset+limit].tolist(),
                'total': len(data['X']), 'schema': tabular_schema(dataset)['features']}


@tabular_router.post('/{dataset}/models/{name}/score')
async def tabular_score(dataset: str, name: str, batch: PairBatch):
    schema = tabular_schema(dataset)
    x = feature_matrix(batch, schema['feature_count'])
    model = tabular_model(dataset, name)
    scores = model.predict_proba(x)[:, 1]
    threshold = schema['models'][name]['operating_points']['0.01']['threshold_selected_on_training_calibration']
    return {'scores': scores.tolist(), 'threshold': threshold,
            'accepted': (scores >= threshold).tolist(),
            'score_direction': schema['score_direction'],
            'operating_point': 'training_calibrated_far_1pct'}


@router.get('/{split}/samples')
async def paired_samples(split: str, offset: int = Query(0, ge=0),
                         limit: int = Query(128, ge=1, le=256)):
    if split not in {'train', 'dev'}:
        raise HTTPException(403, 'Test data is sealed')
    if split != 'dev':
        raise HTTPException(404, 'Only frozen dev pair export is available')
    with np.load(BEACON / 'dev_pairs.npz', allow_pickle=False) as data:
        end = offset + limit
        return {'features': data['X'][offset:end].tolist(),
                'genuine': data['y'][offset:end].tolist(),
                'identities': data['groups'][offset:end].tolist(),
                'total': len(data['X']), 'schema': 'beacon-pair-33-v1'}


@router.post('/models/{name}/score')
async def paired_score(name: str, batch: PairBatch):
    x = feature_matrix(batch, 33)
    config = configuration()
    if name == 'equal_behavior':
        distances = x[:, :32].mean(axis=1)
        threshold = config['baseline_thresholds']['far_1pct']
    else:
        model = model_for(name)
        spec = config['models'][name]
        distances = -model.decision_function(x[:, spec['columns']])
        threshold = spec['thresholds']['far_1pct']
    return {'scores': (-distances).tolist(), 'threshold': -threshold,
            'accepted': (distances <= threshold).tolist(),
            'score_direction': 'higher_is_more_genuine',
            'operating_point': 'training_calibrated_far_1pct',
            'context_is_identity_proof': False}
