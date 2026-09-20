"""Read-only, paired policy analysis and typing-only synthetic diagnostic."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / '.worktrees/typing-pointer/experiments/results'
source = RESULTS / 'typing-pointer-neural-dev-v2.json'
data = json.loads(source.read_text())
old = {r['id']: r for r in data['results']['legacy']}
new = {r['id']: r for r in data['results']['neural']}
assert old.keys() == new.keys()
owners = sorted({k.split('-')[0] for k in old})
assert len(owners) == 15
out = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'accounts': len(owners),
       'method': 'Two-sided exact account-level paired sign-flip test; all 2^15 label swaps. '
                 'Conditional on the artificial mapping and exchangeability under the null. '
                 'Repeated scenarios within an account stay together. Exploratory, not population confirmation.'}
rng = np.random.default_rng(20260920)
signs = np.asarray(list(itertools.product([-1, 1], repeat=len(owners))))
draws = rng.integers(0, len(owners), size=(20000, len(owners)))
for metric, genuine in [('far', False), ('frr', True)]:
    improved = worsened = 0
    delta = []
    for owner in owners:
        errors = []
        for k, r in old.items():
            if k.split('-')[0] != owner or r['genuine'] != genuine: continue
            a = (r['final'] != 'allow') if genuine else (r['final'] == 'allow')
            b = (new[k]['final'] != 'allow') if genuine else (new[k]['final'] == 'allow')
            improved += int(a and not b)
            worsened += int(b and not a)
            errors.append(int(b)-int(a))
        delta.append(float(np.mean(errors)))
    delta = np.asarray(delta)
    observed = float(delta.mean())
    null = (signs * delta).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(observed)-1e-12))
    out[metric] = {'new_minus_old_pp':100*observed,'improved_cases':improved,'worsened_cases':worsened,
                   'exact_cluster_sign_flip_p':p,
                   'account_bootstrap_percentile_95_pp':(100*np.quantile(delta[draws].mean(axis=1), [.025,.975])).tolist(),
                   'account_error_rate_differences':delta.tolist()}

typing_source=RESULTS/'typing-pointer-dev-v1.json'
typing=json.loads(typing_source.read_text())['attempts']
diagnostics={}
for name,allowed in [('ordinary_desktop', {'genuine_same_device','genuine_new_device','genuine_drift',
                                         'impostor_same_device','impostor_new_device'}),
                     ('all_available_typing', {r['scenario'] for r in typing})]:
    cases=[r for r in typing if r['typing_available'] and r['scenario'] in allowed]
    assert all('RBF typing model' in ' '.join(r['typing_model']) for r in cases)
    g=[r for r in cases if r['genuine']];i=[r for r in cases if not r['genuine']]
    fa=sum(r['typing_score']<=r['typing_threshold'] for r in i)
    fr=sum(r['typing_score']>r['typing_threshold'] for r in g)
    diagnostics[name]={'genuine_n':len(g),'impostor_n':len(i),'false_accepts':fa,'false_rejects':fr,
                       'far':fa/len(i),'frr':fr/len(g),
                       'scope':'Typing score <= its model threshold only; excludes bot/device/challenge decisions. '
                               'Post-hoc diagnostic on the existing synthetic scenarios, not a new independent benchmark.'}
out['typing_diagnostics']=diagnostics
out['typing_source_sha256']=hashlib.sha256(typing_source.read_bytes()).hexdigest()
(HERE/'paired-comparison-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k in ('far','frr','typing_diagnostics')},indent=2))
