#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/.research-deps:$PWD/code/bioprint:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TMPDIR="$PWD/.research-tmp"
export PYTHONPYCACHEPREFIX="$PWD/.research-tmp/pycache"
export MPLCONFIGDIR="$PWD/.research-tmp/matplotlib"
export TORCH_HOME="$PWD/.research-tmp/torch"
mkdir -p "$TMPDIR"
exec "${BIGIDEA_PYTHON:-$HOME/miniconda3/envs/bigidea/bin/python}" "$@"
