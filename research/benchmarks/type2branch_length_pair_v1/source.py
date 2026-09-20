"""Run the two frozen TRAIN-only arms sequentially within the CPU/RAM budget."""
import json
from pathlib import Path
import subprocess
import sys
from type2branch_continuity import ROOT, sha


def main():
    out=ROOT/'research/benchmarks/type2branch_length_pair_v1';out.mkdir(exist_ok=False)
    files=[ROOT/'scripts'/n for n in ['type2branch_length_pair.py','type2branch_length_train.py',
        'type2branch_length_training_core.py','type2branch_scoring.py','keystroke_benchmark.py']]
    plan={'arms':['full','mixed'],'execution':'Sequential fresh processes, twoCPUthreads each',
        'updates_per_arm':300,'starting_optimizer_updates':300,'total_new_optimizer_updates':600,
        'selection':'LowestmeanTRAINselectionFRRat1%over25/50/75/100,thenmeanEER,thenearliestupdate; exactarmtiepreferfull',
        'scope':'No calibration/DEV/test arrays or scores; no currentDEVpolicychange',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    for arm in plan['arms']:
        print(json.dumps({'starting_arm':arm}),flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/type2branch_length_train.py'),'--arm',arm],cwd=ROOT,check=True)
    from type2branch_length_training_core import selection_key
    reports={arm:json.loads((ROOT/f'research/benchmarks/type2branch_length_train_{arm}_v1/report.json').read_text()) for arm in plan['arms']}
    if any(r['status']!='paired_arm_complete' or r['updates']!=600 for r in reports.values()):
        raise ValueError('Both equal-budget arms must complete')
    if reports['full']['batch_stream_sha256']!=reports['mixed']['batch_stream_sha256']:
        raise ValueError('Paired input batch streams differed')
    import numpy as np
    baseline_error=0.
    with np.load(ROOT/'research/benchmarks/type2branch_length_train_full_v1/epoch_002_selection.npz',allow_pickle=False) as a, np.load(ROOT/'research/benchmarks/type2branch_length_train_mixed_v1/epoch_002_selection.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['subjects'],b['subjects'])
        np.testing.assert_array_equal(a['labels'],b['labels'])
        for length in [25,50,75,100]:
            baseline_error=max(baseline_error,float(np.max(np.abs(a[f'scores_{length}']-b[f'scores_{length}']))))
    if baseline_error>1e-6:raise ValueError('Paired warm-start baseline inference mismatch')
    selected=min(plan['arms'],key=lambda arm:selection_key(reports[arm]['best_checkpoint']))
    result={'status':'paired_train_comparison_complete','selected_arm':selected,
        'selected_checkpoint':reports[selected]['best_checkpoint'],
        'arm_best':{arm:r['best_checkpoint'] for arm,r in reports.items()},
        'paired_batch_stream_sha256':reports['full']['batch_stream_sha256'],
        'max_warm_start_score_error':baseline_error,
        'limitations':['TRAINselection only, not generalization performance',
            'Oracle selection thresholds are not deployed thresholds','OriginalfailedDEVprotocol remainsunchanged',
            'Feature truncation is not exact shortened raw synthesis']}
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
