"""Replay public CMU train/dev events through BioPrint's real enrollment/login API.

Never opens the live database. Uses a repository-local temporary database. This
is frozen production-pipeline validation, not training or a claim of multimodal
biometric accuracy. No synthetic pointer/device identities are attached.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
from eval.strict_cmu import load_partition, distances
from fastapi.testclient import TestClient


def sample(names, values):
    d = dict(zip(names, values))
    codes = [name[2:] for name in names if name.startswith('H.')]
    events, t = [], 500.
    for index, token in enumerate(codes):
        code = token.split('#')[0]
        events += [{'code': code, 'type': 'down', 't': t, 'field': 'password', 'trusted': True},
                   {'code': code, 'type': 'up', 't': t + d[f'H.{token}'], 'field': 'password', 'trusted': True}]
        if index + 1 < len(codes):
            t += d[f'DD.{token}.{codes[index+1]}']
    return {'keystrokes': sorted(events, key=lambda e: e['t']), 'pointer': [], 'env': {},
            'meta': {'submit_via': 'enter'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, default=51)
    args = parser.parse_args()
    out = ROOT / 'research/benchmarks/cmu/production_api_replay.json'
    if out.exists():
        raise SystemExit('Existing replay report found; inspect logs before rerunning')
    names, train = load_partition('train')
    _, dev = load_partition('dev')
    frozen = json.loads((out.parent / 'frozen_models.json').read_text())
    started = time.monotonic()
    (ROOT / '.research-tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cmu-api-', dir=ROOT / '.research-tmp') as temp:
        os.environ['BIOPRINT_DB'] = str(Path(temp) / 'evaluation.db')
        import db
        import server
        importlib.reload(db)
        importlib.reload(server)
        attempts, enrollment_errors = [], []
        owners = sorted(train)[:args.subjects]
        password = '.tie5Roanl'
        with TestClient(server.app) as client:
            for position, owner in enumerate(owners):
                user = f'public-cmu-{owner}'
                reply = client.post('/api/register', json={'username': user, 'password': password})
                reply.raise_for_status()
                rows = train[owner]['X'][train[owner]['session'] == 1]
                for index in [10] + list(range(10)):
                    response = client.post('/api/enroll', json={'username': user, 'password': password,
                                                               'sample': sample(names, rows[index])})
                    response.raise_for_status()
                    payload = response.json()
                    if not payload.get('accepted'):
                        enrollment_errors.append({'owner': owner, 'index': index, 'response': payload})
                if not payload.get('enrolled'):
                    continue
                other = sorted(dev)[(sorted(dev).index(owner) + 1) % len(dev)]
                for actual in [owner, other]:
                    for session in [5, 6]:
                        vector = dev[actual]['X'][dev[actual]['session'] == session][0]
                        response = client.post('/api/login', json={'username': user, 'password': password,
                                                                  'sample': sample(names, vector)})
                        response.raise_for_status()
                        body = response.json()
                        signals = {row['name']: row for row in body.get('signals', [])}
                        expected = float(distances(frozen['models']['baseline'][owner], vector[None])[0])
                        observed = signals.get('keystroke', {}).get('score')
                        if observed is not None and abs(observed - expected) > 1e-5:
                            raise AssertionError(f'Feature/API parity mismatch {owner}: {observed} vs {expected}')
                        attempts.append({'owner': owner, 'actual': actual, 'session': session,
                                         'genuine': actual == owner, 'decision': body['decision'],
                                         'keystroke_score': observed, 'offline_score': expected,
                                         'bot_flagged': signals.get('bot', {}).get('flagged'),
                                         'latency_ms': body.get('latency_ms')})
                print(f'{position+1}/{len(owners)} owners replayed', flush=True)
        genuine = [r for r in attempts if r['genuine']]
        impostor = [r for r in attempts if not r['genuine']]
        report = {'source': 'CMU sessions1 train enrollment; sessions5–6 dev requests; test7–8 untouched',
                  'model': 'current shipped production scorer, not experimental SVM',
                  'selection': 'first repetition per dev session; next sorted identity as impostor; fixed before replay',
                  'subjects': len(owners), 'attempts': attempts, 'enrollment_errors': enrollment_errors,
                  'frr': sum(r['decision'] != 'allow' for r in genuine) / max(len(genuine), 1),
                  'far': sum(r['decision'] == 'allow' for r in impostor) / max(len(impostor), 1),
                  'genuine_count': len(genuine), 'impostor_count': len(impostor),
                  'elapsed_seconds': time.monotonic() - started,
                  'limitations': ['Sparse deterministic API sample, not full offline ROC.',
                                 'No public pointer/device pairing available for CMU; those signals are missing, not fabricated.',
                                 'Real endpoints and event reconstruction exercised; input trusted flag is client-declared.',
                                 'No live database was opened or modified.']}
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'attempts'}, indent=2))


if __name__ == '__main__':
    main()
