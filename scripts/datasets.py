#!/usr/bin/env python3
"""Discover optional datasets; download only when explicitly requested."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {
 'cmu': ('https://www.cs.cmu.edu/~keystroke/', 'code/bioprint/eval/data/DSL-StrongPasswordData.csv', None),
 'typing-demo': ('https://www.cs.cmu.edu/~keystroke/', 'research/benchmarks/login_combinations_v3', 'experiments/build_cohort.py'),
 'sapimouse': ('https://www.ms.sapientia.ro/~manyi/sapimouse/', 'data/benchmarks/sapimouse', 'scripts/pointer_sapimouse_prepare.py'),
 'balabit': ('https://github.com/balabit/Mouse-Dynamics-Challenge', 'data/benchmarks/balabit', 'scripts/pointer_download.py'),
 'device': ('https://github.com/Spirals-Team/FPStalker', 'datasets/fpstalker', 'scripts/device_download.py'),
 'beacon': ('https://huggingface.co/datasets/beacon-gui/BEACON-Dataset', 'datasets/beacon', None),
 'hmog': ('https://www.cs.wm.edu/~qyang/hmog.html', 'datasets/hmog', None),
 'keyrecs': ('https://doi.org/10.5281/zenodo.7886743', 'research/benchmarks/data/keyrecs', None),
 'other': ('docs/DATASETS.md', 'datasets/', None),
}


def cmu():
    target = ROOT / CATALOG['cmu'][1]
    expected = json.loads((ROOT / 'experiments/cohort_manifest.json').read_text())['source_sha256']
    if target.exists():
        data = target.read_bytes()
    else:
        print('Downloading public CMU timing dataset ...', flush=True)
        with urllib.request.urlopen(CATALOG['cmu'][0] + 'DSL-StrongPasswordData.csv', timeout=90) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise RuntimeError('Unexpected dataset size; download stopped.')
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('CMU source checksum differs from the frozen experiment; existing files were not changed.')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    print(f'CMU verified: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=CATALOG)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    url, destination, script = CATALOG[args.dataset]
    print(f'Source: {url}\nDestination: {ROOT / destination}')
    if not args.download:
        print('No download performed. Add --download for supported automatic sources. See docs/DATASETS.md.')
        return 0
    if args.dataset in ('cmu', 'typing-demo'):
        cmu()
    if args.dataset == 'typing-demo' and (ROOT / destination).exists():
        print('Cohort directory exists; preserving it. The demonstration checks its manifest.')
        return 0
    if script:
        env = dict(os.environ, BIOPRINT_RESEARCH_ROOT=str(ROOT))
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    elif args.dataset != 'cmu':
        print('This source needs manual preparation/terms review. Follow docs/DATASETS.md; no files downloaded.')
        return 2
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Dataset setup failed: {exc}\nSee docs/DATASETS.md for source and placement instructions.')
