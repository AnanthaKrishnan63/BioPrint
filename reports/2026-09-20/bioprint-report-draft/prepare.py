"""Build report tables and a complete companion ledger from saved evidence only."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


summary = json.loads(read('reports/2026-09-20/evidence/summary_v11.json'))
rows = summary['rows']
def get(dataset, model):
    return next(r for r in rows if r['dataset'] == dataset and r['model'] == model)
def rates(r):
    return ' / '.join(f"{100*r[k]:.2f}" for k in ('far', 'frr', 'eer'))

pairs = [
    ('CMU, 10 enrollments', 'cmu-account-selection', 'baseline', 'account_selected'),
    ('CMU, 100 enrollments', 'cmu100', 'baseline', 'svm-1-1'),
    ('KeyRecs fixed text', 'keyrecs-fixed', 'reference_logistic', 'train_selected'),
    ('KeyRecs free text', 'keyrecs-free', 'reference_logistic', 'train_selected'),
    ('SapiMouse, five blocks', 'pointer_sapimouse_optimized', 'scaled_manhattan@5', 'cosine@5'),
    ('FPStalker browser linkage', 'device', 'equal_attribute_agreement', 'hist_gradient_boosting'),
    ('DELBOT bot geometry', 'delbot', 'bioprint_pointer_rules_only', 'geometric_random_forest'),
    ('Stroop/Flanker cognition', 'cognitive', 'condition_index_slope', 'full_rt_accuracy_profile'),
    ('TSI touch landing', 'touch_tsi', 'landing_baseline', 'selected'),
    ('Balabit pointer', 'pointer', 'existing_scaled_manhattan', 'extra_1'),
]
table = []
for name, dataset, old, new in pairs:
    a, b = get(dataset, old), get(dataset, new)
    table.append(f'{name} & {rates(a)} & {rates(b)} & {100*(a["eer"]-b["eer"]):+.2f} \\\\')
(HERE/'dataset-table.tex').write_text('\n'.join(table)+'\n\\bottomrule\n')
with (HERE/'eer-plot.dat').open('w') as f:
    f.write('index baseline candidate\n')
    for i, (_, dataset, old, new) in enumerate(pairs[:5]):
        f.write(f'{i} {100*get(dataset,old)["eer"]:.6f} {100*get(dataset,new)["eer"]:.6f}\n')

ledger = []
for r in rows:
    ledger.append(dict(scope='dataset research',dataset=r['dataset'],model=r['model'],
                       far_percent=100*r['far'],frr_percent=100*r['frr'],eer_percent=100*r['eer'],
                       aggregation=r['eer_aggregation'],source='reports/2026-09-20/evidence/'+r['source']))

# Keep all general-matcher rows, including controls and cross-dataset transfer.
for filename in ('RESULTS.md','ALIGNED_RESULTS.md'):
    path='reports/2026-09-20/evidence/general-typing/'+filename
    for line in read(path).splitlines():
        if not line.startswith('|'): continue
        cells=[s.strip() for s in line.strip('|').split('|')]
        if len(cells)<7 or not all('%' in cells[i] for i in (2,3,4)): continue
        ledger.append(dict(scope='general typing '+filename,dataset=cells[0],model=cells[1],
                           far_percent=float(cells[2].rstrip('%')),frr_percent=float(cells[3].rstrip('%')),
                           eer_percent=float(cells[4].rstrip('%')),false_accepts=cells[5],false_rejects=cells[6],
                           aggregation='pooled; rounded source table',source=path))

comparison_path='experiments/results/comparison.json'
comparison=json.loads(read(comparison_path))
policy_table=[]
for r in comparison['rows']:
    ledger.append(dict(scope='synthetic policy '+r['split'],dataset='shared artificial cohort',model=r['mode'],
                       far_percent=100*r['far'],frr_percent=100*r['frr'],eer_percent='',
                       false_accepts=f'{r["false_accepts"]}/{r["impostor_n"]}',
                       false_rejects=f'{r["false_rejects"]}/{r["genuine_n"]}',
                       aggregation='final scenario outcomes; EER undefined',source=comparison_path))
    if r['split']=='dev':
        name={'login-control':'Control','typing-stepup':'Typing repetition','typing-pointer':'Typing + statistical pointer','typing-keypad':'Typing + keypad'}[r['mode']]
        policy_table.append(f'{name} & {r["false_accepts"]}/90 & {100*r["far"]:.2f} & {r["false_rejects"]}/60 & {100*r["frr"]:.2f} \\\\')
for version in ('v1','v2'):
    path=f'experiments/results/typing-pointer-neural-dev-{version}.json'
    data=json.loads(read(path))
    for name,r in data['overall'].items():
        ledger.append(dict(scope='synthetic neural '+version,dataset='shared artificial DEV cohort',model=name,
                           far_percent=100*r['far'],frr_percent=100*r['frr'],eer_percent='',
                           false_accepts=f'{r["false_accepts"]}/{r["impostor_n"]}',
                           false_rejects=f'{r["false_rejects"]}/{r["genuine_n"]}',
                           aggregation='final scenario outcomes; EER undefined',source=path))
    if version=='v2':
        r=data['overall']['neural']
        assert r['false_accepts']==9 and r['false_rejects']==19
        policy_table.append(r'\textbf{Typing + trained pointer} & 9/90 & \textbf{10.00} & 19/60 & \textbf{31.67} \\')
(HERE/'policy-table.tex').write_text('\n'.join(policy_table)+'\n\\bottomrule\n')
tradeoff_path='experiments/results/typing-pointer-neural-tradeoff.json'
tradeoff=json.loads(read(tradeoff_path))
for target,r in tradeoff['threshold_tradeoff'].items():
    ledger.append(dict(scope='synthetic pointer threshold counterfactual',dataset='shared artificial DEV cohort',
                       model='source calibration FAR target '+target,
                       far_percent=100*r['far'],frr_percent=100*r['frr'],eer_percent='',
                       false_accepts=f'{r["false_accepts"]}/{r["impostor_n"]}',
                       false_rejects=f'{r["false_rejects"]}/{r["genuine_n"]}',
                       aggregation='fixed score-level counterfactual, not full API rerun',source=tradeoff_path))
for r in summary['infeasible_experiments']:
    ledger.append(dict(scope='infeasible study',dataset=r['experiment'],model=r['status'],
                       far_percent='',frr_percent='',eer_percent='',aggregation='recognition metrics unavailable',
                       source='reports/2026-09-20/evidence/'+r['source']))
ledger.append(dict(scope='schema audit only',dataset='BrainRun',model='101 games; 982 gestures; 7/8 TRAIN users',
                   aggregation='recognition metrics unavailable',source='reports/2026-09-20/evidence/brainrun_train_schema_v2/report.json'))
ledger.append(dict(scope='genuine-only bot audit',dataset='CMU',model='production timing rules',
                   frr_percent=100*15/5100,false_rejects='15/5100',aggregation='genuine false flags; no bot-positive FAR/EER',
                   source='reports/2026-09-20/evidence/bot_rules/results.json'))
read('code/bioprint/experiment.json')
analysis=json.loads(read('reports/2026-09-20/bioprint-report-draft/paired-comparison-analysis.json'))
for name,r in analysis['typing_diagnostics'].items():
    ledger.append(dict(scope='post-hoc synthetic typing-only diagnostic',dataset=name,model='compatible RBF SVM at model threshold',
                       far_percent=100*r['far'],frr_percent=100*r['frr'],eer_percent='',
                       false_accepts=f'{r["false_accepts"]}/{r["impostor_n"]}',
                       false_rejects=f'{r["false_rejects"]}/{r["genuine_n"]}',aggregation=r['scope'],
                       source='experiments/results/typing-pointer-dev-v1.json'))
read('experiments/typing_demo/demo.py')
read('reports/2026-09-20/bioprint-report-draft/typing-demo-results.json')
for path in ('reports/2026-09-20/DATASET_RESULTS.md',
             'experiments/NEURAL_POINTER_RESULTS.md',
             'code/bioprint/EXPERIMENT.md'):
    read(path)
fields=['scope','dataset','model','far_percent','frr_percent','eer_percent','false_accepts','false_rejects','aggregation','source']
with (HERE/'complete-results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(ledger)
(HERE/'evidence-manifest.json').write_text(json.dumps({'sources':SOURCES,'ledger_rows':len(ledger),
    'source_reports_sha256':summary['source_sha256'],
    'infeasible_experiments':summary['infeasible_experiments'],
    'notes':['Repeated controls are preserved, not independent experiments.',
             'All rates refer to source protocols, not the latest live app.',
             'General matcher rates are rounded in their source Markdown tables.']},indent=2)+'\n')
print(f'Prepared {len(ledger)} ledger rows and source-derived tables.')
