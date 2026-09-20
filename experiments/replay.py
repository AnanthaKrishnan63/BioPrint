"""Replay the identical frozen cohort through a branch's actual login APIs.

No listening server; fresh temporary database; actual session and challenge checks.
Only the challenge RNG is seeded to issue the frozen layout/target for each run.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cohort', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--split', choices=['train', 'calibration', 'dev'], required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists(): raise SystemExit('Refusing to overwrite a completed replay')
manifest = json.loads((args.cohort/'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((args.cohort/name).read_bytes()).hexdigest() == expected, name
data = json.loads(gzip.decompress((args.cohort/'cohort.json.gz').read_bytes()))
appdir = Path(__file__).resolve().parents[1]/'code/bioprint'
sys.path.insert(0, str(appdir))
source_paths = [p for p in appdir.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.js') and 'tests' not in p.parts and '__pycache__' not in p.parts]
source_hashes = {str(p.relative_to(appdir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
os.environ['BIOPRINT_BACKGROUND'] = str((args.cohort/'background.json').resolve())
os.environ['BIOPRINT_BACKGROUND_SHA256'] = manifest['files']['background.json']
from fastapi.testclient import TestClient


def summarize(attempts):
    genuine = [a for a in attempts if a['genuine']]
    impostor = [a for a in attempts if not a['genuine']]
    fa = sum(a['final']=='allow' for a in impostor)
    fr = sum(a['final']!='allow' for a in genuine)
    return dict(genuine_n=len(genuine), impostor_n=len(impostor), false_accepts=fa, false_rejects=fr,
                far=fa/len(impostor) if impostor else None, frr=fr/len(genuine) if genuine else None,
                stepup_n=sum(a['initial'] in ('step_up','keypad') for a in attempts),
                genuine_stepup_n=sum(a['initial'] in ('step_up','keypad') for a in genuine),
                typing_available_n=sum(a['typing_available'] for a in attempts),
                nonterminal_n=sum(a['final'] not in ('allow','block') for a in attempts),
                eer=None, eer_reason='Branching final policy has no prespecified single scalar EER score')


started = time.monotonic()
attempts = []
with tempfile.TemporaryDirectory(prefix='login-replay-', dir=os.environ['TMPDIR']) as tmp:
    os.environ['BIOPRINT_DB'] = str(Path(tmp)/'isolated.db')
    import server
    config = json.loads((appdir/'experiment.json').read_text())
    with TestClient(server.app) as client:
        def post(path, body):
            r = client.post(path, json=body)
            r.raise_for_status()
            return r.json()

        def issued(user, original):
            run = copy.deepcopy(original)
            server.random.seed(run.pop('_challenge_seed'))
            challenge = post('/api/keypad/challenge', {'username':user})
            assert challenge['layout']==run['layout'] and challenge['target']==run['target']
            run['challenge_id'] = challenge['id']
            return run

        people = [p for p in data['people'] if p['role']==args.split]
        for p in people:
            creds = dict(username=p['id'], password=data['password'])
            post('/api/register', creds)
            for sample in p['enrollment']:
                r = post('/api/enroll', dict(**creds, sample=sample))
                assert r['accepted'], r
            assert r['enrolled'], r
            for run in p['keypad_enrollment']:
                r = post('/api/enroll/keypad', dict(**creds, runs=[issued(p['id'],run)]))
                assert r['accepted'], r
            assert r['enrolled'], r
        print(f"{config['mode']}: enrolled {len(people)} {args.split} profiles", flush=True)
        for a in [x for x in data['attempts'] if x['role']==args.split]:
            client.cookies.clear()
            creds = dict(username=a['owner'], password=data['password'])
            r = post('/api/login', dict(**creds, sample=a['sample']))
            initial = r['decision']
            initial_signals = r['signals']
            latencies = [r['latency_ms']]
            if initial=='step_up':
                r = post('/api/login/stepup', dict(**creds, samples=a['stepup']))
                latencies.append(r['latency_ms'])
            if r['decision']=='keypad':
                runs = [issued(a['owner'],x) for x in a['runs']]
                r = post('/api/login/keypad', dict(**creds, runs=runs))
                latencies.append(r['latency_ms'])
            session = client.get('/api/session')
            assert (session.status_code==200) == (r['decision']=='allow'), (a['id'],r,session.text)
            ks = next((x for x in initial_signals if x['name']=='keystroke'), {})
            attempts.append(dict(id=a['id'], genuine=a['genuine'], scenario=a['scenario'], initial=initial,
                                 final=r['decision'], typing_available=ks.get('available',False),
                                 typing_model=ks.get('reasons',[])[:1], typing_score=ks.get('score'),
                                 typing_threshold=ks.get('threshold'), signals=r['signals'],
                                 server_scoring_ms=sum(latencies)))
        # Inspect no production database; this connection is the temporary DB only.
    result = dict(mode=config['mode'], split=args.split, config=config, manifest=manifest,
                  source_hashes=source_hashes,
                  overall=summarize(attempts),
                  by_scenario={s:summarize([a for a in attempts if a['scenario']==s]) for s in sorted({a['scenario'] for a in attempts})},
                  attempts=attempts, elapsed_seconds=time.monotonic()-started,
                  listener_started=False, database='fresh temporary database; deleted after replay',
                  interpretation='Synthetic screening only; simulated target trajectories, devices and trust fields')
assert all(hashlib.sha256((appdir/name).read_bytes()).hexdigest() == expected for name, expected in source_hashes.items()), 'Source changed during replay'
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result['overall']), flush=True)
