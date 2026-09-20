#!/usr/bin/env bash
set -euo pipefail
checkout_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export BIOPRINT_RESEARCH_ROOT="${BIOPRINT_RESEARCH_ROOT:-$checkout_dir}"
exec bash "$checkout_dir/scripts/research.sh" "$checkout_dir/experiments/typing_demo/demo.py" "$@"
