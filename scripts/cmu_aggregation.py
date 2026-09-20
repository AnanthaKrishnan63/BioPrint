"""Predeclared observation-length experiment; train-only aggregation selection.

Uses the already training-selected ten-enrollment SVM. No model is selected from
dev. Aggregation never mixes actual users or sessions. Longer evidence has a
real interaction cost; this is not single-login performance.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/bioprint'), str(ROOT / '.research-deps')]
from eval.strict_cmu import load_partition, distances, eer, threshold_for, rates
OUT = ROOT / 'research/benchmarks/cmu_aggregation'


def score_groups(models, records, sessions, count, reducer):
    outputs = {}
    for owner, model in models.items():
        genuine, impostor = [], []
        for actual, data in records.items():
            for session in sessions:
                scores = distances(model, data['X'][data['session'] == session])
                n = len(scores) // count
                reduced = getattr(np, reducer)(scores[:n*count].reshape(n, count), axis=1)
                (genuine if actual == owner else impostor).extend(reduced)
        outputs[owner] = np.array(genuine), np.array(impostor)
    return outputs


def main():
    if (OUT / 'dev_results.json').exists():
        raise SystemExit('Already evaluated; inspect logs rather than repeat')
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT/'research/benchmarks/cmu/frozen_models.json').read_text())
    _, train = load_partition('train')
    frozen = {'counts': [1, 4, 8], 'selection': 'training sessions2–3', 'calibration': 'training session4',
              'cost': 'count consecutive password repetitions, never mixed across users/sessions', 'models': {}}
    # Freeze all settings and thresholds before loading dev in this experiment.
    for method, models in source['models'].items():
        frozen['models'][method] = {}
        for count in frozen['counts']:
            selection = {}
            for reducer in ['mean', 'median']:
                scores = score_groups(models, train, [2, 3], count, reducer)
                selection[reducer] = float(np.mean([eer(g, i) for g, i in scores.values()]))
            chosen = min(selection, key=selection.get)
            calibration = score_groups(models, train, [4], count, chosen)
            thresholds = {u: {key: threshold_for(g, i, target) for key,target in [('far_1pct', .01), ('far_5pct', .05)]}
                          for u,(g,i) in calibration.items()}
            frozen['models'][method][str(count)] = {'reducer': chosen, 'selection_eer': selection, 'thresholds': thresholds}
            print(method, count, chosen, selection, flush=True)
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    _, dev = load_partition('dev')
    result = {'protocol': {k:v for k,v in frozen.items() if k!='models'}, 'results': {}}
    for method, models in source['models'].items():
        result['results'][method] = {}
        for count_text, config in frozen['models'][method].items():
            scores = score_groups(models, dev, [5, 6], int(count_text), config['reducer'])
            per = {u:{'eer':eer(g,i), **{key:rates(g,i,t) for key,t in config['thresholds'][u].items()}}
                   for u,(g,i) in scores.items()}
            report = {'macro_eer':float(np.mean([r['eer'] for r in per.values()])),
                      'operating_points':{key:{rate:float(np.mean([r[key][rate] for r in per.values()])) for rate in ['far','frr']}
                                          for key in ['far_1pct','far_5pct']},
                      'genuine_groups_per_user':len(next(iter(scores.values()))[0]), 'per_user':per}
            result['results'][method][count_text] = report
            print(method,count_text,{k:v for k,v in report.items() if k!='per_user'},flush=True)
    (OUT/'dev_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
