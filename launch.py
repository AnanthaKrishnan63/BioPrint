#!/usr/bin/env python3
"""Run the final BioPrint app on loopback only; no dataset downloads."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import socket
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'code' / 'bioprint'


def configure(data_dir):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12: conda env create -f environment.yml; conda activate bigidea')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '2')
    for package in ('fastapi', 'uvicorn', 'numpy', 'sklearn', 'torch'):
        try:
            importlib.import_module(package)
        except ImportError as exc:
            raise RuntimeError('Missing dependency. Activate bigidea and run: python -m pip install -r requirements.txt') from exc
    manifest = json.loads((ROOT / 'models' / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = ROOT / 'models' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or damaged model asset: {path}. Restore models/ from the repository.')
    sys.path.insert(0, str(APP))
    os.environ['BIOPRINT_DB'] = str(data_dir / 'bioprint.db')
    os.environ['BIOPRINT_POINTER_ENCODER'] = str(ROOT / 'models' / 'pointer-encoder.pt')
    os.environ['BIOPRINT_BACKGROUND'] = str(ROOT / 'models' / 'typing-background.json')
    os.environ['BIOPRINT_BACKGROUND_SHA256'] = manifest['sha256']['typing-background.json']
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '2')
    from engine import experiment
    if experiment.MODE != 'typing-pointer':
        raise RuntimeError('The main release requires the typing-pointer policy.')
    from pointer_neural import encoder
    import numpy as np
    result = encoder().model(__import__('torch').zeros((1, 2, 128))).detach().numpy()
    if result.shape != (1, 128) or not np.isfinite(result).all():
        raise RuntimeError('Pointer encoder failed its startup check.')
    bank = experiment.background()
    if bank is None or not bank.get('fit') or not bank.get('calibration'):
        raise RuntimeError('Typing background/calibration bank is incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate dependencies and model assets without serving')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runtime', help='Local database and session-key directory')
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    try:
        configure(data_dir)
        if args.check:
            print('Ready: typing-pointer policy, trained pointer encoder and compatible typing bank. CPU / localhost:8000.')
            return 0
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', 8000))
            except OSError as exc:
                raise RuntimeError('Port 8000 is occupied. Stop the existing service yourself, then rerun launch.py.') from exc
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_file = data_dir / 'session.key'
        try:
            fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'w') as handle:
                handle.write(secrets.token_hex(32))
        secret = key_file.read_text().strip()
        if len(secret) < 32:
            raise RuntimeError(f'Invalid session key at {key_file}; restore it from your local backup.')
        os.environ['BIOPRINT_SECRET'] = secret
        os.umask(0o077)
        print(f'BioPrint: http://localhost:8000\nLocal data: {data_dir}\nStop with Ctrl-C.', flush=True)
        import uvicorn
        uvicorn.run('server:app', host='127.0.0.1', port=8000, proxy_headers=False)
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'BioPrint startup: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
