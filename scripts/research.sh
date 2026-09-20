#!/usr/bin/env bash
# Run optional research using the activated Conda environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${BIGIDEA_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo 'Activate the bigidea Conda environment first.' >&2
  exit 1
fi
export PYTHONPATH="$PWD/code/bioprint:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TMPDIR="$PWD/.research-tmp"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export TORCH_HOME="$TMPDIR/torch"
mkdir -p "$TMPDIR"
exec "$python_bin" "$@"
