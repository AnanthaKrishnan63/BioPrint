"""Actual neural-pointer API replay on an explicitly synthetic cross-dataset cohort."""
import copy
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zlib
import numpy as np

ROOT = Path(os.environ['BIOPRINT_RESEARCH_ROOT'])
BRANCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRANCH / 'code/bioprint'))
COHORT = ROOT / 'research/benchmarks/login_combinations_v3'
SAPI = ROOT / 'data/benchmarks/sapimouse'
OPT = ROOT / 'research/benchmarks/pointer_sapimouse_optimized'
parser = argparse.ArgumentParser()
parser.add_argument('--version', choices=['v1', 'v2'], default='v2')
args = parser.parse_args()
OUT = BRANCH / f'experiments/results/typing-pointer-neural-dev-{args.version}.json'
if OUT.exists():
    raise SystemExit('Refusing to overwrite a completed experiment')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(rows):
    g = [r for r in rows if r['genuine']]
    i = [r for r in rows if not r['genuine']]
    fa = sum(r['final'] == 'allow' for r in i)
    fr = sum(r['final'] != 'allow' for r in g)
    return {'genuine_n': len(g), 'impostor_n': len(i), 'false_accepts': fa, 'false_rejects': fr,
            'far': fa / len(i) if i else None, 'frr': fr / len(g) if g else None,
            'challenge_n': sum(r['initial'] in ('keypad', 'step_up') for r in rows),
            'neural_n': sum(r.get('neural_used', False) for r in rows),
            'incomplete_n': sum(r['final'] == 'incomplete' for r in rows), 'eer': None}


manifest = json.loads((COHORT / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert digest(COHORT / name) == expected
data = json.loads(gzip.decompress((COHORT / 'cohort.json.gz').read_bytes()))
people = sorted((p for p in data['people'] if p['role'] == 'dev'), key=lambda p: p['id'])
attempts = [a for a in data['attempts'] if a['role'] == 'dev']
source_manifest = json.loads((SAPI / 'split_manifest.json').read_text())
source_entries = {e['path']: e for e in source_manifest['files']}
with np.load(OPT / 'cosine@5_dev_scores.npz', allow_pickle=False) as a:
    score_matrix, actual, owners, groups = (a[k].copy() for k in ('scores', 'true_user', 'users', 'sessions'))
rng = np.random.default_rng(20260920)
mapping = dict(zip([p['id'] for p in people], rng.permutation(owners)[:len(people)].tolist()))
following = dict(zip(mapping, list(mapping.values())[1:] + list(mapping.values())[:1]))
pairings = {}
for a in attempts:
    source_user = mapping[a['owner']] if a['genuine'] or a['scenario'] == 'keypad_match_attack' else following[a['owner']]
    col = int(rng.choice(np.flatnonzero(actual == source_user)))
    pairings[a['id']] = {'owner': mapping[a['owner']], 'actual': source_user, 'column': col,
                         'score': float(score_matrix[list(owners).index(mapping[a['owner']]), col])}
plan = {'protocol_sha256': digest(BRANCH / 'experiments/NEURAL_POINTER_PROTOCOL.md'),
        'version': args.version, 'stationary_coordinate_fix': args.version == 'v2',
        'server_sha256': digest(BRANCH / 'code/bioprint/server.py'),
        'pointer_module_sha256': digest(BRANCH / 'code/bioprint/pointer_neural.py'),
        'script_sha256': digest(Path(__file__)),
        'cohort_sha256': manifest['files']['cohort.json.gz'], 'source_scores_sha256': digest(OPT / 'cosine@5_dev_scores.npz'),
        'mapping': mapping, 'pairings': pairings, 'synthetic_cross_dataset_people': True,
        'production_database_accessed': False, 'test_accessed': False}
(BRANCH / f'experiments/results/typing-pointer-neural-{args.version}-plan.json').write_text(json.dumps(plan, indent=2))
raw_cache = {}


def source_events(column):
    group = str(groups[column])
    entry = source_entries[group]
    assert entry['split'] == 'dev' and entry['role'] == 'dev_probe'
    if group not in raw_cache:
        content = (SAPI / group).read_bytes()
        assert zlib.crc32(content) == entry['crc']
        raw_cache[group] = list(csv.DictReader(content.decode().splitlines()))
    ordinal = int(np.count_nonzero(groups[:column] == group))
    rows = raw_cache[group][ordinal * 640:ordinal * 640 + 641]
    assert len(rows) == 641
    t0 = float(rows[0]['client timestamp'])
    return [{'type': 'move', 't': float(r['client timestamp']) - t0, 'x': float(r['x']), 'y': float(r['y']),
             'pointer_type': 'mouse', 'trusted': True} for r in rows]


os.environ['BIOPRINT_BACKGROUND'] = str(COHORT / 'background.json')
os.environ['BIOPRINT_BACKGROUND_SHA256'] = manifest['files']['background.json']
os.environ['BIOPRINT_POINTER_ENCODER'] = str(ROOT / 'research/benchmarks/pointer_sapimouse/encoder.torchscript.pt')
from fastapi.testclient import TestClient
started = time.perf_counter()
results = {'legacy': [], 'neural': []}
with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'], prefix='neural-pointer-replay-') as tmp:
    live = Path(tmp) / 'simulation.db'
    os.environ['BIOPRINT_DB'] = str(live)
    import server
    import db
    import pointer_neural as neural
    with TestClient(server.app) as client:
        def post(path, body):
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        def issued(username, original):
            run = copy.deepcopy(original)
            server.random.seed(run.pop('_challenge_seed'))
            challenge = post('/api/keypad/challenge', {'username': username})
            assert challenge['layout'] == run['layout'] and challenge['target'] == run['target']
            run['challenge_id'] = challenge['id']
            return run

        for p in people:
            creds = {'username': p['id'], 'password': data['password']}
            post('/api/register', creds)
            for sample in p['enrollment']:
                assert post('/api/enroll', {**creds, 'sample': sample})['accepted']
            for run in p['keypad_enrollment']:
                assert post('/api/enroll/keypad', {**creds, 'runs': [issued(p['id'], run)]})['accepted']
        legacy = Path(tmp) / 'legacy.db'
        shutil.copyfile(live, legacy)
        with db.connect() as conn:
            for p in people:
                with np.load(OPT / f'cosine_{mapping[p["id"]]}.npz', allow_pickle=False) as a:
                    center = a['center'].tolist()
                model = {'center': center, 'threshold': neural.THRESHOLD, 'encoder_sha256': neural.ENCODER_SHA256,
                         'blocks_per_decision': 5, 'source': 'frozen public TRAIN-support profile; synthetic mapping'}
                uid = conn.execute('SELECT id FROM users WHERE username=?', (p['id'],)).fetchone()[0]
                conn.execute('INSERT INTO neural_pointer_profiles VALUES (?,?,?,?)', (uid, 'mouse', json.dumps(model), db.now()))
        learned = Path(tmp) / 'learned.db'
        shutil.copyfile(live, learned)
        print('Enrolled identical synthetic typing/keypad profiles; loaded separate public pointer support profiles', flush=True)
        for mode, snapshot in [('legacy', legacy), ('neural', learned)]:
            for attempt in attempts:
                shutil.copyfile(snapshot, live)
                client.cookies.clear()
                creds = {'username': attempt['owner'], 'password': data['password']}
                reply = post('/api/login', {**creds, 'sample': attempt['sample']})
                initial = reply['decision']
                used = False
                pointer_score = None
                error = None
                if initial == 'step_up':
                    reply = post('/api/login/stepup', {**creds, 'samples': attempt['stepup']})
                if reply['decision'] == 'keypad':
                    if any(s['name'] == 'pointer_neural' for s in reply['signals']):
                        used = True
                        token = post('/api/pointer/challenge', {**creds, 'kind': 'verify'})['id']
                        events = source_events(pairings[attempt['id']]['column'])
                        # Simulate elapsed capture time in this disposable DB only.
                        with db.connect() as conn:
                            conn.execute('UPDATE neural_pointer_challenges SET expires=? WHERE id=?',
                                         (time.time() + 180 - (events[-1]['t'] / 1000 + 1), token))
                        response = client.post('/api/pointer/capture', json={**creds, 'kind': 'verify', 'challenge_id': token, 'events': events})
                        if response.status_code == 422:
                            error = response.json()['detail']
                            reply = {'decision': 'incomplete'}
                        else:
                            response.raise_for_status()
                            reply = response.json()
                            pointer_score = 1 - next(s['score'] for s in reply['signals'] if s['name'] == 'pointer_neural')
                            assert abs(pointer_score - pairings[attempt['id']]['score']) < 2e-6
                    else:
                        reply = post('/api/login/keypad', {**creds, 'runs': [issued(attempt['owner'], r) for r in attempt['runs']]})
                assert (client.get('/api/session').status_code == 200) == (reply['decision'] == 'allow')
                results[mode].append({'id': attempt['id'], 'scenario': attempt['scenario'], 'genuine': attempt['genuine'],
                    'initial': initial, 'final': reply['decision'], 'neural_used': used,
                    'pointer_similarity': pointer_score, 'incomplete_reason': error})
            print(mode, json.dumps(stats(results[mode])), flush=True)
original = json.loads((BRANCH / 'experiments/results/typing-pointer-dev-v1.json').read_text())
old = {a['id']: a for a in original['attempts']}
differences = [a['id'] for a in results['legacy'] if a['initial'] != old[a['id']]['initial'] or a['final'] != old[a['id']]['final']]
report = {'plan': plan, 'results': results, 'overall': {k: stats(v) for k, v in results.items()},
          'by_scenario': {k: {s: stats([a for a in rows if a['scenario'] == s]) for s in sorted({a['scenario'] for a in rows})} for k, rows in results.items()},
          'old_stored_result_differences': differences, 'elapsed_seconds': time.perf_counter() - started,
          'encoder_sha256': neural.ENCODER_SHA256, 'threshold': neural.THRESHOLD,
          'interpretation': 'Synthetic cross-dataset composition through actual model and APIs; not real-person multimodal accuracy'}
OUT.write_text(json.dumps(report, indent=2))
print('Completed', OUT, flush=True)
