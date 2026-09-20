"""Freeze real CMU typing plus explicitly simulated target-task identities.

No claim of matching people across datasets. CMU sessions 7–8 are never parsed.
Use bigidea; BIOPRINT_RESEARCH_ROOT points at the original research workspace.
"""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(os.environ['BIOPRINT_RESEARCH_ROOT']).resolve()
sys.path[:0] = [str(ROOT / 'code/bioprint'), str(ROOT / 'code/bioprint/tests')]
import numpy as np
from eval.strict_cmu import load_partition, SOURCE
from helpers import make_run
from contracts import KEYPAD_BLANK
from test_bot import human_env

OUT = ROOT / 'research/benchmarks/login_combinations_v3'
OUT.mkdir(exist_ok=False)
SEED = 20260920
names, records = load_partition('train')
_, dev = load_partition('dev')
ids = sorted(records, key=lambda s: hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest())
bank_ids, fit_ids, cal_ids, dev_ids = ids[:12], ids[12:24], ids[24:36], ids[36:]
roles = {s: role for role, group in [('train', fit_ids), ('calibration', cal_ids), ('dev', dev_ids)] for s in group}
bank = {'role': 'train_background', 'names': names, 'source_ids': bank_ids,
        'fit': [x.tolist() for s in bank_ids for x in records[s]['X'][records[s]['session'] == 1][:10]],
        'calibration': [x.tolist() for s in bank_ids for x in records[s]['X'][records[s]['session'] == 4][:10]]}
(OUT / 'background.json').write_text(json.dumps(bank, sort_keys=True))
rng = random.Random(SEED)
params = {s: dict(interval_ms=rng.uniform(350, 850), interval_sd=90., hold_ms=rng.uniform(65, 135),
                  curve=rng.uniform(.02, .2), land_sd=rng.uniform(4, 14), hover_ms=rng.uniform(45, 120)) for s in roles}


def environment(s, mobile=False):
    d = human_env(fonts=['Arial', f'SyntheticFont{s}'], webgl_vendor='NVIDIA Corporation',
                  webgl_renderer='NVIDIA GeForce RTX 3060', timezone='Asia/Kolkata')
    if mobile:
        d.update(ua='Mozilla/5.0 (Android 14; Mobile) Firefox/155.0', ua_mobile=True,
                 platform='Linux armv8l', max_touch_points=5, plugins=0,
                 screen_width=412, screen_height=915, inner_width=412, inner_height=800,
                 outer_width=412, outer_height=915, webgl_vendor='Qualcomm', webgl_renderer='Adreno 730')
    return d


def sample(x, env, virtual=False):
    values = dict(zip(names, x))
    codes = [name[2:] for name in names if name.startswith('H.')]
    events, t = [], 500.
    for i, token in enumerate(codes):
        code = token.split('#')[0]
        events.extend([dict(code=code, type='down', t=t, field='password', trusted=True),
                       dict(code=code, type='up', t=t+values[f'H.{token}'], field='password', trusted=True)])
        if i+1 < len(codes):
            t += values[f'DD.{token}.{codes[i+1]}']
    if virtual:
        events = [dict(code='Unidentified', type=kind, t=500.+i*150., field='password', trusted=True)
                  for i in range(len(codes)) for kind in ('down','up')]
    return dict(keystrokes=sorted(events, key=lambda e:e['t']), pointer=[], env=copy.deepcopy(env), meta=dict(submit_via='enter'))


def run(s, seed, env, mobile=False, drift=1.):
    r = random.Random(seed)
    layout = list(range(5))+[KEYPAD_BLANK]
    r.shuffle(layout)
    target = []
    while len(target) < 6:
        digit = r.randrange(5)
        if len(target)>=2 and target[-1]==digit and target[-2]==digit:
            continue
        target.append(digit)
    p = dict(params[s]); p['interval_ms'] *= drift
    result = make_run(seed=seed+100000, device='touch' if mobile else 'mouse', layout=layout,
                      target=target, env=copy.deepcopy(env), **p)
    result['_challenge_seed'] = seed
    return result


people, attempts = [], []
for ix, s in enumerate(roles):
    env = environment(s)
    enroll = records[s]['X'][records[s]['session'] == 1][:11]
    people.append(dict(id=s, role=roles[s], enrollment=[sample(x, env) for x in enroll],
                       keypad_enrollment=[run(s, 1000+ix*10+j, env) for j in range(6)]))
    group = [x for x in roles if roles[x] == roles[s]]
    other = group[(group.index(s)+1) % len(group)]
    session = {'train':2, 'calibration':4, 'dev':5}[roles[s]]
    data = dev if roles[s]=='dev' else records
    scenarios = [
        ('genuine_same_device', s, s, False, False, False, 1.),
        ('genuine_new_device', s, s, True, False, False, 1.),
        ('genuine_mobile', s, s, True, True, False, 1.),
        ('genuine_drift', s, s, False, False, False, 1.25),
        ('impostor_same_device', other, other, False, False, False, 1.),
        ('impostor_new_device', other, other, True, False, False, 1.),
        ('impostor_mobile', other, other, True, True, False, 1.),
        ('typing_match_attack', s, other, False, False, False, 1.),
        ('keypad_match_attack', other, s, False, False, False, 1.),
        ('scripted_attack', other, other, False, False, True, 1.),
    ]
    for j,(scenario, typist, target_person, changed, mobile, bot, drift) in enumerate(scenarios):
        env = environment(other if changed else s, mobile)
        if changed and not mobile:env.update(timezone='Europe/London', fonts=['Calibri', 'Segoe UI'])
        xs = data[typist]['X'][data[typist]['session']==session][j*4:j*4+4]
        samples = [sample(x, env, mobile) for x in xs]
        if bot:
            for item in samples:
                for e in item['keystrokes']: e['trusted'] = False
        attempts.append(dict(id=f'{s}-{scenario}', owner=s, role=roles[s], genuine=scenario.startswith('genuine'),
                             scenario=scenario, sample=samples[0], stepup=samples[1:],
                             runs=[run(target_person, 1000000+ix*100+j*2+k, env, mobile, drift) for k in range(2)]))
cohort = dict(version=1, seed=SEED, password='.tie5Roanl', people=people, attempts=attempts,
              limitations=['Typing is real CMU; keypad, motor, device and trust fields are simulated.',
                           'No measured cross-modal human correlation; no independent all-feature EER.',
                           'Previously explored CMU; synthetic DEV is not pristine final test.',
                           'Mobile enrollment assumes prior physical-keyboard enrollment; mobile-only signup is not solved.'])
for a in attempts:
    assert sum(e['type']=='down' for e in a['sample']['keystrokes']) == len(cohort['password'])
    for target_run in a['runs']:
        assert sorted(target_run['layout']) == [KEYPAD_BLANK,0,1,2,3,4]
raw = json.dumps(cohort, sort_keys=True, separators=(',',':')).encode()
(OUT/'cohort.json.gz').write_bytes(gzip.compress(raw, mtime=0))
manifest = dict(seed=SEED, people=len(people), attempts=len(attempts), roles={r:sum(p['role']==r for p in people) for r in roles.values()},
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                files={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['background.json','cohort.json.gz']},
                limitations=cohort['limitations'], test_observations_parsed=False)
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
for f in OUT.iterdir(): f.chmod(0o444)
print(json.dumps(manifest, indent=2))
