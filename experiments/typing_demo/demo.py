"""One-command, real typing-model/API demonstration. No live database or listener."""
import argparse
import copy
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import tempfile
import webbrowser

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--no-open', action='store_true', help='Do not open the generated HTML in a browser')
args=parser.parse_args()
HERE=Path(__file__).resolve().parent
BRANCH=HERE.parents[1]
ROOT=Path(os.environ['BIOPRINT_RESEARCH_ROOT']).resolve()
COHORT=ROOT/'research/benchmarks/login_combinations_v3'
manifest=json.loads((COHORT/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((COHORT/name).read_bytes()).hexdigest()==expected, name
data=json.loads(gzip.decompress((COHORT/'cohort.json.gz').read_bytes()))
evidence=BRANCH/'experiments/results/typing-pointer-dev-v1.json'
reference=json.loads(evidence.read_text())['attempts']

# Illustrative examples are selected transparently from existing DEV outcomes.
# No model parameter or threshold is changed to manufacture a demonstration.
def choose(genuine, match, scenario):
    candidates=[r for r in reference if r['genuine']==genuine and r['scenario']==scenario
                and r['typing_available'] and (r['typing_score']<=r['typing_threshold'])==match]
    assert candidates, (genuine,match,scenario)
    return sorted(candidates,key=lambda r:r['id'])[0]
selected=[('True accept',choose(True,True,'genuine_same_device')),
          ('True reject',choose(False,False,'impostor_same_device')),
          ('False reject',choose(True,False,'genuine_same_device')),
          ('False accept',choose(False,True,'impostor_same_device')),
          ('Owner-like typing attack',choose(False,True,'typing_match_attack'))]
attempts={r['id']:r for r in data['attempts']}
people={p['id']:p for p in data['people']}
owners=sorted({attempts[r['id']]['owner'] for _,r in selected})
appdir=BRANCH/'code/bioprint'
sys.path.insert(0,str(appdir))
os.environ['BIOPRINT_BACKGROUND']=str(COHORT/'background.json')
os.environ['BIOPRINT_BACKGROUND_SHA256']=manifest['files']['background.json']
from fastapi.testclient import TestClient

print('BioPrint | Trained typing demonstration',flush=True)
print('Public dataset password: '+data['password'],flush=True)
print('Fresh temporary database; actual enrollment, RBF SVM fitting and login API; no network listener.',flush=True)
results=[]
with tempfile.TemporaryDirectory(prefix='typing-demo-',dir=os.environ['TMPDIR']) as tmp:
    os.environ['BIOPRINT_DB']=str(Path(tmp)/'demo.db')
    import server
    with TestClient(server.app) as client:
        def post(path,body):
            response=client.post(path,json=body);response.raise_for_status();return response.json()
        for owner in owners:
            print('Enrolling synthetic account '+owner+' ...',flush=True)
            creds={'username':owner,'password':data['password']}
            post('/api/register',creds)
            for sample in people[owner]['enrollment']:
                result=post('/api/enroll',dict(**creds,sample=sample));assert result['accepted']
            assert result['enrolled']
            for original in people[owner]['keypad_enrollment']:
                run=copy.deepcopy(original)
                server.random.seed(run.pop('_challenge_seed'))
                challenge=post('/api/keypad/challenge',{'username':owner})
                assert challenge['layout']==run['layout'] and challenge['target']==run['target']
                run['challenge_id']=challenge['id']
                result=post('/api/enroll/keypad',dict(**creds,runs=[run]));assert result['accepted']
            assert result['enrolled']
        for label,saved in selected:
            case=attempts[saved['id']]
            client.cookies.clear()
            reply=post('/api/login',{'username':case['owner'],'password':data['password'],'sample':case['sample']})
            signal=next(s for s in reply['signals'] if s['name']=='keystroke')
            assert signal['available'] and any('RBF typing model' in reason for reason in signal['reasons'])
            assert abs(signal['score']-saved['typing_score'])<1e-6
            assert reply['decision']==saved['initial'], (saved['id'],reply['decision'],saved['initial'])
            session=client.get('/api/session')
            assert (session.status_code==200)==(reply['decision']=='allow')
            matched=signal['score']<=signal['threshold']
            observed=('True ' if matched==case['genuine'] else 'False ')+('accept' if matched else 'reject')
            assert label==observed or label=='Owner-like typing attack'
            result={'label':label,'case_id':case['id'],'ground_truth':'owner' if case['genuine'] else 'impostor',
                    'typing_outcome':observed,'score':signal['score'],'model_limit':signal['threshold'],
                    'direct_admission_limit':.75*signal['threshold'],'login_decision':reply['decision'],
                    'session_issued':session.status_code==200,'source_model':signal['reasons'][0]}
            results.append(result)
            print(f"{label:24s} | score {signal['score']:.4f} / {signal['threshold']:.2f} | login: {reply['decision']} | session: {result['session_issued']}",flush=True)

outdir=BRANCH/'experiments/runtime/typing-demo'
outdir.mkdir(parents=True,exist_ok=True)
payload={'password_is_public_dataset_text':data['password'],'examples':results,
         'cohort_sha256':manifest['files']['cohort.json.gz'],
         'reference_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
         'interpretation':'Outcome-selected demonstration, not an unbiased accuracy benchmark. '
                          'First four labels describe the typing classifier at limit 1.0. '
                          'The full login separately uses the stricter direct limit 0.75, bot/device checks and step-up. '
                          'The fifth example intentionally supplies owner-like typing to an impostor; it is not a natural impostor trace.',
         'real_api':True,'listener_started':False,'temporary_database_deleted':True}
(outdir/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
cards=[]
for r in results:
    text='This is an intentionally owner-like typing attack.' if r['label']=='Owner-like typing attack' else 'Ground truth: '+r['ground_truth']+'.'
    if r['label']=='False accept': text+=' The typing model matches, but the stricter app policy requests a keypad check; no session is issued.'
    cards.append(f'''<article><p class="eyebrow">{html.escape(r['case_id'])}</p><h2>{html.escape(r['label'])}</h2>
      <p>{text}</p><div class="score">{r['score']:.4f} <small>/ {r['model_limit']:.2f} model limit</small></div>
      <p>Typing classifier: <b>{html.escape(r['typing_outcome'])}</b></p>
      <p>Actual login decision: <b>{html.escape(r['login_decision'])}</b><br>Session issued: <b>{str(r['session_issued']).lower()}</b></p></article>''')
document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BioPrint | Trained typing demonstration</title><style>
body{font:17px/1.6 system-ui,sans-serif;background:#f2f6f8;color:#17334d;margin:0;padding:40px}main{max-width:1150px;margin:auto}
h1{font-size:38px;line-height:1.15}h2{font-size:23px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#087f8c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:20px}article{background:white;border:1px solid #d9e4e9;border-top:4px solid #087f8c;border-radius:12px;padding:22px}
.score{font-size:28px}small{font-size:14px}aside{background:#e2eff0;padding:20px;margin:25px 0;border-radius:10px}code{background:white;padding:3px 7px;border-radius:4px}
</style><main><p class="eyebrow">Actual model • Actual login API • Isolated data</p><h1>Four outcomes.<br>One trained typing model.</h1>
<p>A radial basis function support vector machine compares password timing with the enrolled account holder.
All cases below were freshly scored through BioPrint, not assigned canned verdicts.</p>
<aside>The public dataset password is <code>'''+html.escape(data['password'])+'''</code>. Lower scores are a closer match.
The model limit is <b>1.00</b>; direct login requires <b>0.75</b> on a familiar device. A typing match is therefore not automatically a successful login.</aside>
<div class="grid">'''+''.join(cards)+'''</div><aside>These examples were selected to illustrate the outcomes, not to estimate accuracy.
Typing traces come from the frozen CMU-based synthetic cohort. The owner-like attack deliberately substitutes owner behavior.
No real user database was used and no network service was started.</aside></main></html>'''
page=outdir/'index.html';page.write_text(document)
print('\nVerified all five actual API outcomes. Open '+str(page),flush=True)
if not args.no_open: webbrowser.open(page.as_uri())
