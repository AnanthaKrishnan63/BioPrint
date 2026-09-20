"""TRAIN-prefix-only sequence-count feasibility; never decode timing/key fields."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/benchmarks/data/keyrecs'
OUT = ROOT / 'research/benchmarks/type2branch_keyrecs_eligibility_v1'


def count_train_prefixes(lines, active):
    header = next(lines).split(',', 2)
    if header[:2] != ['participant', 'session']:
        raise ValueError('Unexpected source prefix')
    counts = Counter()
    for line in lines:
        participant, separator, rest = line.partition(',')
        session, separator2, opaque = rest.partition(',')
        if not separator or not separator2:
            raise ValueError('Invalid metadata prefix')
        # Opaque measurement suffix is never decoded, including TRAIN rows.
        if participant in active and session == '1':
            counts[participant] += 1
    return counts


def cardinalities(count):
    windows = count // 100
    first, second = windows // 2, windows * 3 // 4
    return {'rows': count, 'full_100_row_windows': windows,
            'fit_windows': first, 'selection_windows': second-first,
            'calibration_windows': windows-second, 'tail_rows': count % 100}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    manifest_path = DATA/'split-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    active, sealed = set(manifest['active_subjects']), set(manifest['sealed_test_subjects'])
    if active & sealed:
        raise ValueError('Split identity overlap')
    source = DATA/'free-text.csv'
    expected = manifest['files']['free-text.csv']['sha256']
    plan = {'scope': 'TRAIN row counts only, prefix metadata; no observation values decoded',
            'source_sha256': expected, 'manifest_sha256': sha(manifest_path),
            'script_sha256': sha(Path(__file__)), 'sequence_rows': 100, 'N': 15, 'K': 10,
            'candidate_train_partition': 'chronological full windows first50%, next25%, last25%',
            'not_authorized': ['DEV/test observation parsing', 'model selection', 'training']}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    if sha(source) != expected:
        raise ValueError('Source changed after original split')
    with source.open() as stream:
        counts = count_train_prefixes(stream, active)
    if sum(counts.values()) != manifest['files']['free-text.csv']['rows']['train']:
        raise ValueError('TRAIN count differs from sealed manifest')
    rows = {subject: cardinalities(counts[subject]) for subject in sorted(active)}
    fit = [s for s,r in rows.items() if r['fit_windows'] >= 15]
    select = [s for s,r in rows.items() if r['selection_windows'] >= 15]
    both = sorted(set(fit) & set(select))
    report = {'status': 'metadata_count_audit_complete', 'active_identities': len(active),
              'train_rows': sum(counts.values()), 'per_identity': rows,
              'fit_N15_eligible': fit, 'selection_N15_eligible': select,
              'both_N15_eligible': both,
              'fit_K10_batch_possible_by_count': len(fit) >= 10,
              'selection_K10_batch_possible_by_count': len(select) >= 10,
              'same_cohort_K10_both_possible_by_count': len(both) >= 10,
              'timing_or_key_values_decoded': False,
              'limitations': ['Rows are digraph records, not proven contiguous raw events',
                             'Count feasibility does not establish quality or sequence independence',
                             'No identity omitted from audit; no cohort selected for training',
                             'Chronological protocol candidate is a small-data adaptation']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_identity'}))


if __name__ == '__main__':
    main()
