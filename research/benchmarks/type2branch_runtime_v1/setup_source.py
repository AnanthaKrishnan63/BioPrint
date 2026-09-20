"""Install an isolated CPU reference candidate inside the repository."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/type2branch_runtime_v1'
TARGET = ROOT / '.research-type2branch-deps'


def main():
    OUT.mkdir(exist_ok=False)
    TARGET.mkdir(exist_ok=False)
    (OUT / 'setup_source.py').write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ, TMPDIR=str(ROOT / '.research-tmp'),
               PYTHONPYCACHEPREFIX=str(ROOT / '.research-tmp/pycache'))
    commands = [
        [sys.executable, '-m', 'pip', '--isolated', 'install', '--index-url',
         'https://pypi.org/simple', '--no-cache-dir', '--only-binary=:all:',
         '--ignore-installed', '--target', str(TARGET), '--report',
         str(OUT / 'tensorflow_install.json'), 'tensorflow-cpu==2.16.1'],
        [sys.executable, '-m', 'pip', '--isolated', 'install', '--index-url',
         'https://pypi.org/simple', '--no-cache-dir', '--only-binary=:all:',
         '--no-deps', '--target', str(TARGET), '--report',
         str(OUT / 'tf_keras_install.json'), 'tf-keras==2.16.0'],
    ]
    (OUT / 'plan.json').write_text(json.dumps({'commands': commands,
        'purpose': 'Candidate CPU runtime, not claimed original author environment',
        'legacy_keras': True, 'addons_replacement_not_validated': True,
        'dataset_access': False}, indent=2))
    for i, command in enumerate(commands):
        with (OUT / f'install_{i}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Installation stage {i} failed: {result.returncode}; inspect saved log')
    (OUT / 'installed.json').write_text(json.dumps({'status': 'installed_not_yet_executed',
                                                  'target': str(TARGET)}, indent=2))
    print('CPU TensorFlow and legacy Keras candidate installed; execution validation pending')


if __name__ == '__main__':
    main()
