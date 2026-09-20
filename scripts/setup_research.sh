#!/usr/bin/env bash
# Optional setup; existing frozen experiments do not need reinstallation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
research_python="${BIGIDEA_PYTHON:-$HOME/miniconda3/envs/bigidea/bin/python}"
if [[ ! -x "$research_python" ]]; then
  printf '%s\n' 'Create the bigidea environment first, or set BIGIDEA_PYTHON to its Python executable.' >&2
  exit 1
fi
mkdir -p .research-tmp/pip-cache .research-deps
export TMPDIR="$PWD/.research-tmp"
export PIP_CACHE_DIR="$PWD/.research-tmp/pip-cache"
export PYTHONPYCACHEPREFIX="$PWD/.research-tmp/pycache"
"$research_python" -m pip install --target "$PWD/.research-deps" \
  -r research/benchmarks/requirements-research.txt
"$research_python" -m pip install --target "$PWD/.research-deps" --no-deps \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
