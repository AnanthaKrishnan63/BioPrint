"""Audit research dataset disk sizes using metadata only; never open recordings."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENTS = ['datasets', 'data/benchmarks', 'research/benchmarks/data']


def apparent_bytes(directory, allow_internal_aliases=False):
    total = directory.stat().st_size
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            entry = Path(parent) / name
            if entry.is_symlink():
                if allow_internal_aliases and entry.resolve(strict=True).is_relative_to(directory.resolve()):
                    # Pytest creates "current" aliases to already counted
                    # fixture directories. Count the link, never traverse it.
                    total += entry.lstat().st_size
                    continue
                raise ValueError(f'Dataset symlink requires an explicit budget audit: {entry}')
            total += entry.stat().st_size
    return total


def source_document_files(directory, already_counted=()):
    """Include downloaded schema documents/receipts using filesystem metadata."""
    counted = set(already_counted)
    result = {}
    if not directory.exists():
        return result
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Source-document symlink requires an explicit budget audit: {path}')
        if path.is_file() and path not in counted:
            result[path] = path.stat().st_size
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directories = [ROOT / 'code/bioprint/eval/data']
    for name in PARENTS:
        parent = ROOT / name
        if parent.exists():
            directories.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
    rows = {str(p.relative_to(ROOT)): apparent_bytes(p) for p in directories if p.exists()}
    if 'datasets/arithmetic' in rows:
        rows['datasets/arithmetic'] += rows.pop('datasets/arithmetic_metadata', 0)
    if 'datasets/hmog' in rows:
        # Discovery includes HMOG archive metadata (and a small BB-MAS note).
        # Charge all of it to HMOG conservatively instead of splitting a source.
        rows['datasets/hmog'] += rows.pop('datasets/joint_feasibility_metadata', 0)
    # Conservatively include derived feature/score arrays stored beside models.
    # Some npz files are profile parameters, so this may overcount dataset bytes.
    derived = {}
    for path in (ROOT / 'research/benchmarks').rglob('*'):
        if not path.is_file() or path.suffix not in {'.npz', '.npy', '.csv', '.jsonl'}:
            continue
        if any(path.is_relative_to(directory) for directory in directories):
            continue
        rel = str(path.relative_to(ROOT / 'research/benchmarks'))
        if rel.startswith(('cmu',)):
            owner = 'code/bioprint/eval/data'
        elif rel.startswith(('results/keyrecs', 'results/typenet')):
            owner = 'research/benchmarks/data/keyrecs'
        elif rel.startswith('pointer_sapimouse'):
            owner = 'data/benchmarks/sapimouse'
        elif rel.startswith('pointer'):
            owner = 'data/benchmarks/balabit'
        elif rel.startswith('device'):
            owner = 'datasets/fpstalker'
        else:
            owner = next((f'datasets/{name}' for name in ['beacon', 'cognitive', 'delbot', 'touch_tsi', 'hmog', 'arithmetic']
                          if rel.startswith(name)), 'other_derived_reports')
        size = path.stat().st_size
        derived[str(path.relative_to(ROOT))] = size
        rows[owner] = rows.get(owner, 0) + size
    # Source schema PDFs, images, text and acquisition receipts were previously
    # reported as a separate manual adjustment. Include them automatically.
    documents = source_document_files(ROOT / 'research/benchmarks/source_documents',
                                      (ROOT / p for p in derived))
    for path, size in documents.items():
        owner = 'datasets/hmog' if path.name.startswith('hmog_') else 'other_derived_reports'
        rows[owner] = rows.get(owner, 0) + size
    # Test/replay fixtures may retain public-derived data outside dataset roots.
    # Count every other temporary file conservatively; package/cache files are
    # not datasets. Assigning all fixtures to the largest source below gives a
    # safe per-dataset upper bound without opening fixture contents.
    temporary = {}
    temporary_root = ROOT / '.research-tmp'
    if temporary_root.exists():
        for path in temporary_root.iterdir():
            if path.name in {'pip-cache', 'pycache'}:
                continue
            if path.is_symlink():
                raise ValueError(f'Temporary symlink requires an explicit budget audit: {path}')
            temporary[str(path.relative_to(ROOT))] = apparent_bytes(path, allow_internal_aliases=True) if path.is_dir() else path.stat().st_size
    temporary_total = sum(temporary.values())
    total = sum(rows.values()) + temporary_total
    largest_with_all_temporary = max(
        (size for source, size in rows.items() if source != 'other_derived_reports'), default=0
    ) + temporary_total + rows.get('other_derived_reports', 0)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'apparent_bytes_by_dataset': rows, 'total_bytes': total,
              'derived_arrays_included': derived,
              'source_documents_included': {str(p.relative_to(ROOT)): n for p, n in documents.items()},
              'temporary_fixtures_included': temporary,
              'temporary_fixture_bytes': temporary_total,
              'largest_dataset_upper_bound_with_all_temporary_bytes': largest_with_all_temporary,
              'per_dataset_limit_bytes': 500_000_000,
              'total_limit_bytes': 5_000_000_000,
              'within_budget': total < 5_000_000_000 and largest_with_all_temporary < 500_000_000,
              'method': 'Filesystem metadata only; no contents read, including sealed recordings.',
              'scope': 'Archives, extracted data, metadata, derived arrays, downloaded source documents/receipts and temporary fixtures; npz profile files conservatively included. Excludes joblib/torch model weights, Python dependencies and package/bytecode caches.'}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if not report['within_budget']:
        raise SystemExit('Dataset budget exceeded')


if __name__ == '__main__':
    main()
