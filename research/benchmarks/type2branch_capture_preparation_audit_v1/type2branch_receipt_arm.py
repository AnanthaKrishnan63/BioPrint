"""Independently verify completed TRAIN-arm score summaries and receipt files."""
import argparse
import json
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_length_training_core import selection_key
from keystroke_benchmark import curve_metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=['full', 'mixed'], required=True)
    arm = parser.parse_args().arm
    run = ROOT/f'research/benchmarks/type2branch_length_train_{arm}_v1'
    report = json.loads((run/'report.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    if report['status'] != 'paired_arm_complete' or report['arm'] != arm or report['updates'] != 600:
        raise ValueError('Completed named arm required')
    for group in ['source_sha256', 'input_sha256']:
        for name, expected in plan[group].items():
            if sha(ROOT/name) != expected: raise ValueError('Frozen arm dependency changed')
    records = report['checkpoints']
    if [r['epoch'] for r in records] != [2, 3, 4, 5]: raise ValueError('Missing checkpoint record')
    if min(records, key=selection_key) != report['best_checkpoint']: raise ValueError('Selection mismatch')
    receipts = {}
    for record in records:
        epoch = record['epoch']
        if record['updates'] != (epoch+1)*100: raise ValueError('Update count mismatch')
        prefix = run/f'epoch_{epoch:03d}_selection.npz'
        with np.load(prefix, allow_pickle=False) as data:
            if data['subjects'].shape != (16,) or len(set(data['subjects'])) != 16:
                raise ValueError('Expected16selection identities')
            np.testing.assert_array_equal(data['labels'], np.repeat(np.arange(16), 10))
            recomputed = {}
            for length in [25, 50, 75, 100]:
                scores = data[f'scores_{length}']; labels = data['labels']
                if scores.shape != (160,16) or not np.isfinite(scores).all():
                    raise ValueError('Expected finite full selection claim matrix')
                g = scores[np.arange(160), labels]
                i = scores[np.arange(16)[None,:] != labels[:,None]]
                recomputed[str(length)] = curve_metrics(g, i)
                for key in ['eer', 'oracle_frr_at_far_1pct']:
                    if abs(recomputed[str(length)][key]-record['per_length'][str(length)][key]) > 1e-12:
                        raise ValueError('Saved selection score summary mismatch')
        for key, field in [('mean_selection_eer','eer'), ('mean_selection_frr_at_1pct','oracle_frr_at_far_1pct')]:
            if abs(np.mean([r[field] for r in recomputed.values()])-record[key]) > 1e-12:
                raise ValueError('Mean selection summary mismatch')
        checkpoint = __import__('pathlib').Path(record['checkpoint']).resolve()
        if checkpoint.parent != run.resolve(): raise ValueError('Checkpoint outside arm')
        pieces = sorted(run.glob(checkpoint.name+'.*'))
        if {p.name for p in pieces} != {checkpoint.name+'.index', checkpoint.name+'.data-00000-of-00001'}:
            raise ValueError('Checkpoint shard set mismatch')
        receipts[str(epoch)] = {'checkpoint': str(checkpoint), 'updates': record['updates'],
            'files': {p.name: sha(p) for p in pieces}, 'selection_scores_sha256': sha(prefix)}
    out = ROOT/f'research/benchmarks/type2branch_length_{arm}_receipt_v1';out.mkdir(exist_ok=False)
    result = {'status': 'completed_arm_scores_verified_and_files_receipted', 'arm': arm,
        'report_sha256': sha(run/'report.json'), 'plan_sha256': sha(run/'plan.json'),
        'checkpoints': receipts, 'best_checkpoint': report['best_checkpoint'],
        'source_sha256': {'scripts/type2branch_receipt_arm.py': sha(ROOT/'scripts/type2branch_receipt_arm.py')},
        'limitations': ['Hashes captured after arm completion, not at checkpoint creation',
                       'Verifies saved score arithmetic; does not rerun encoder inference']}
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'arm': arm, 'checkpoints_verified': len(receipts)}))


if __name__ == '__main__': main()
