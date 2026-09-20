#!/usr/bin/env bash
# Explicit manual launch only. Each worktree gets its own database.
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkout_dir="$(dirname -- "$experiment_dir")"
common_git="$(git -C "$checkout_dir" rev-parse --git-common-dir)"
research_root="${BIOPRINT_RESEARCH_ROOT:-$(dirname -- "$common_git")}"
python_bin="${BIGIDEA_PYTHON:-$HOME/miniconda3/envs/bigidea/bin/python}"
cohort_dir="$research_root/research/benchmarks/login_combinations_v3"
export PYTHONPATH="$checkout_dir/code/bioprint:$research_root/.research-deps"
export TMPDIR="$research_root/.research-tmp"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$experiment_dir/runtime" "$TMPDIR"
export BIOPRINT_DB="$experiment_dir/runtime/bioprint.db"
export BIOPRINT_BACKGROUND="$cohort_dir/background.json"
export BIOPRINT_BACKGROUND_SHA256="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["background.json"])' "$cohort_dir/manifest.json")"
experiment_port="${1:-8005}"
cd "$checkout_dir/code/bioprint"
exec "$python_bin" -m uvicorn server:app --host 127.0.0.1 --port "$experiment_port"
